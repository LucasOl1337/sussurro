"""Aba REUNIAO: grava voce e o audio do PC em duas trilhas, transcreve na GPU com quem
falou o que, e abre reunioes antigas. Toda chamada de widget acontece na thread do Tk."""
import json
import queue
import subprocess
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

import sussurro_meeting as meeting
import sussurro_meeting_audio as rec

BG = '#1b1c20'
FIELD = '#25262b'
FIELD_HOVER = '#2f3036'
BORDER = '#454750'
INK = '#f2f3f3'
INK_2 = '#c3c6c8'
MUTED = '#9ba0a4'
ACCENT = '#f0500a'
ACCENT_HOVER = '#d64708'
GRAPHITE = '#16181a'
WAVE_YOU = '#f07944'
WAVE_PC = '#7aa2c8'
SPEAKER_COLORS = ('#f07944', '#7aa2c8', '#9ccf8a', '#d7a6e8', '#e6c86e', '#6fd0c4', '#e88f8f', '#b0b8c0')
LANG_CHOICES = ('pt', 'en', 'auto')
SPEAKER_CHOICES = ('Automático', '1', '2', '3', '4', '5', '6', '7', '8')


def _select_on_focus(entry):
    """Clicar no campo ja seleciona o nome todo (Ctrl+A do Tk vai pro inicio da linha)."""
    entry.bind('<FocusIn>', lambda _e: entry.after_idle(lambda: entry.select_range(0, 'end')))


class MeetingPanel(ctk.CTkFrame):
    def __init__(self, parent, host):
        super().__init__(parent, fg_color='transparent')
        self.host = host
        self.font = (host.FONT_UI, 12)
        self.events = queue.Queue()
        self.sources = None          # (mic, pc) enquanto a aba esta aberta ou gravando
        self.recording = False
        self.paused = False
        self.elapsed = 0.0           # segundos gravados, sem as pausas
        self.resumed_at = None
        self.started = None
        self.abort = None            # threading.Event da transcricao em curso
        self.folder = None           # reuniao aberta na tela de pronta
        self.manifest = None
        self.player = rec.Player()
        self.line_at = []            # (linha do texto, inicio em ms)
        self.visible = False
        self.closed = False

        self.ready_view = self._build_ready()
        self.work_view = self._build_work()
        self.done_view = self._build_done()
        self._show(self.ready_view)
        self._offer_recovery()
        self.after(50, self._tick)

    # -- construcao ---------------------------------------------------------
    def _button(self, master, text, command, accent=False, width=None):
        opts = dict(text=text, command=command, height=34, corner_radius=8, font=self.font)
        if accent:
            opts.update(fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=GRAPHITE,
                        text_color_disabled=GRAPHITE)
        else:
            opts.update(fg_color=FIELD, hover_color=FIELD_HOVER, text_color=INK_2,
                        border_width=1, border_color=BORDER)
        if width:
            opts['width'] = width
        return ctk.CTkButton(master, **opts)

    def _combo(self, master, values, current, width, command=None):
        box = ctk.CTkComboBox(master, values=list(values), state='readonly', width=width, height=32,
                              font=self.font, fg_color=FIELD, border_color=FIELD_HOVER,
                              button_color=FIELD, button_hover_color=FIELD_HOVER,
                              dropdown_fg_color=FIELD, dropdown_hover_color=FIELD_HOVER,
                              dropdown_text_color=INK, text_color=INK, command=command)
        box.set(current)
        return box

    def _caption(self, master, text):
        return ctk.CTkLabel(master, text=text, text_color=MUTED, anchor='w',
                            font=(self.host.FONT_UI, 10, 'bold'))

    def _build_ready(self):
        view = ctk.CTkFrame(self, fg_color='transparent')
        top = ctk.CTkFrame(view, fg_color='transparent')
        top.pack(fill='x', padx=16, pady=(12, 0))
        top.grid_columnconfigure(0, weight=1)
        self._caption(top, 'NOME DA REUNIÃO').grid(row=0, column=0, sticky='w')
        self._caption(top, 'IDIOMA').grid(row=0, column=1, sticky='w', padx=(8, 0))
        self._caption(top, 'ABRIR ANTERIOR').grid(row=0, column=2, sticky='w', padx=(8, 0))
        # sem textvariable: com ela o CTkEntry nao mostra o placeholder
        self.title_entry = ctk.CTkEntry(top, height=32, font=self.font, fg_color=FIELD,
                                        border_color=FIELD_HOVER, text_color=INK,
                                        placeholder_text='Reunião HH:MM (dá pra mudar durante a call)')
        self.title_entry.grid(row=1, column=0, sticky='ew', pady=(4, 0))
        lang = self.host.settings.get('language', 'pt')
        self.language = self._combo(top, LANG_CHOICES, lang if lang in LANG_CHOICES else 'pt', 80)
        self.language.grid(row=1, column=1, padx=(8, 0), pady=(4, 0))
        self.previous = self._combo(top, ['—'], '—', 220, command=self._open_previous)
        self.previous.grid(row=1, column=2, padx=(8, 0), pady=(4, 0))

        self.recovery = ctk.CTkFrame(view, fg_color=FIELD, corner_radius=8)
        self.recovery_label = ctk.CTkLabel(self.recovery, text='', text_color=INK, font=self.font,
                                           anchor='w', justify='left')
        self.recovery_label.pack(side='left', padx=12, pady=8)
        self._button(self.recovery, 'Descartar', self._discard_recovery, width=90).pack(side='right', padx=(4, 8))
        self._button(self.recovery, 'Salvar como reunião', self._save_recovery, accent=True,
                     width=170).pack(side='right')

        meters = ctk.CTkFrame(view, fg_color=BG, corner_radius=10)
        meters.pack(fill='x', padx=16, pady=(12, 0))
        self.canvases = []
        for label, color in (('VOCÊ (MICROFONE)', WAVE_YOU), ('ÁUDIO DO PC (OS OUTROS)', WAVE_PC)):
            self._caption(meters, label).pack(fill='x', padx=12, pady=(10, 2))
            canvas = tk.Canvas(meters, height=46, bg=FIELD, highlightthickness=0, bd=0)
            canvas.pack(fill='x', padx=12)
            self.canvases.append((canvas, color))
        ctk.CTkFrame(meters, fg_color='transparent', height=10).pack()

        clock = ctk.CTkFrame(view, fg_color='transparent')
        clock.pack(fill='x', padx=16, pady=(12, 0))
        self.clock = ctk.CTkLabel(clock, text='00:00', text_color=INK, font=(self.host.FONT_DISPLAY, 30))
        self.clock.pack(side='left')
        self.state_label = ctk.CTkLabel(clock, text='', text_color=MUTED, font=self.font, anchor='w',
                                        justify='left', wraplength=460)
        self.state_label.pack(side='left', padx=14)

        buttons = ctk.CTkFrame(view, fg_color='transparent')
        buttons.pack(fill='x', padx=16, pady=(12, 0))
        self.start_btn = self._button(buttons, 'Iniciar gravação', self.start, accent=True, width=170)
        self.start_btn.pack(side='left')
        self.pause_btn = self._button(buttons, 'Pausar', self.pause, width=100)
        self.stop_btn = self._button(buttons, 'Parar e transcrever', self.stop, width=170)
        self.voices = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(buttons, text='Separar vozes do mesmo lado', variable=self.voices,
                        font=self.font, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                        text_color=INK_2).pack(side='right')

        imports = ctk.CTkFrame(view, fg_color='transparent')
        imports.pack(fill='x', padx=16, pady=(16, 0))
        self.import_btn = self._button(imports, 'Importar áudio…', self.import_file, width=150)
        self.import_btn.pack(side='left')
        ctk.CTkLabel(imports, text='Falantes:', text_color=MUTED, font=self.font).pack(side='left', padx=(12, 6))
        self.speakers = self._combo(imports, SPEAKER_CHOICES, 'Automático', 120)
        self.speakers.pack(side='left')
        ctk.CTkLabel(view, text='Grava o microfone padrão e o que o PC toca, em trilhas separadas. Nenhum bot entra '
                                'na call e nada sai da máquina. Tudo é transcrito na GPU; o modelo é o do ditado.',
                     text_color=MUTED, font=(self.host.FONT_UI, 11), anchor='w', justify='left',
                     wraplength=680).pack(fill='x', padx=16, pady=(16, 0))
        self.ready_status = ctk.CTkLabel(view, text='', text_color=INK_2, font=self.font, anchor='w',
                                         justify='left', wraplength=680)
        self.ready_status.pack(fill='x', padx=16, pady=(8, 0))
        return view

    def _build_work(self):
        view = ctk.CTkFrame(self, fg_color='transparent')
        self.work_stage = ctk.CTkLabel(view, text='', text_color=INK, font=(self.host.FONT_DISPLAY, 20), anchor='w')
        self.work_stage.pack(fill='x', padx=16, pady=(16, 6))
        self.work_bar = ctk.CTkProgressBar(view, progress_color=ACCENT, fg_color=FIELD, height=10)
        self.work_bar.pack(fill='x', padx=16)
        self.work_bar.set(0)
        self.work_lines = ctk.CTkTextbox(view, fg_color=BG, text_color=INK_2, font=(self.host.FONT_MONO, 12),
                                         wrap='word', state='disabled')
        self.work_lines.pack(fill='both', expand=True, padx=16, pady=12)
        self.cancel_btn = self._button(view, 'Cancelar transcrição', self._cancel_job, width=180)
        self.cancel_btn.pack(anchor='w', padx=16, pady=(0, 12))
        return view

    def _build_done(self):
        view = ctk.CTkFrame(self, fg_color='transparent')
        head = ctk.CTkFrame(view, fg_color='transparent')
        head.pack(fill='x', padx=16, pady=(12, 0))
        head.grid_columnconfigure(0, weight=1)
        self.done_title = tk.StringVar()
        title = ctk.CTkEntry(head, textvariable=self.done_title, height=34, font=(self.host.FONT_DISPLAY, 18),
                             fg_color=FIELD, border_color=FIELD_HOVER, text_color=INK)
        title.grid(row=0, column=0, sticky='ew')
        title.bind('<Return>', lambda _e: self._rename())
        title.bind('<FocusOut>', lambda _e: self._rename())
        _select_on_focus(title)
        self.done_meta = ctk.CTkLabel(head, text='', text_color=MUTED, font=(self.host.FONT_MONO, 11), anchor='w')
        self.done_meta.grid(row=1, column=0, sticky='ew', pady=(4, 0))

        self.names_frame = ctk.CTkFrame(view, fg_color='transparent')
        self.names_frame.pack(fill='x', padx=16, pady=(8, 0))
        self.name_vars = {}

        actions = ctk.CTkFrame(view, fg_color='transparent')
        actions.pack(fill='x', padx=16, pady=(8, 0))
        self._button(actions, 'Copiar transcrição', self._copy, accent=True, width=160).pack(side='left')
        self.play_btn = self._button(actions, 'Ouvir', self._toggle_play, width=80)
        self.play_btn.pack(side='left', padx=(8, 0))
        self._button(actions, 'Abrir pasta', self._open_folder, width=100).pack(side='left', padx=(8, 0))
        self.again_btn = self._button(actions, 'Refazer em', self._again, width=100)
        self.again_btn.pack(side='left', padx=(8, 0))
        self.again_language = self._combo(actions, LANG_CHOICES, 'pt', 70)
        self.again_language.pack(side='left', padx=(4, 0))
        self._button(actions, 'Nova reunião', self._new, width=120).pack(side='right')

        self.transcript = tk.Text(view, bg=BG, fg=INK, insertbackground=INK, relief='flat', wrap='word',
                                  font=(self.host.FONT_UI, 12), padx=12, pady=10, cursor='hand2',
                                  highlightthickness=0, spacing3=8)
        self.transcript.pack(fill='both', expand=True, padx=16, pady=12)
        self.transcript.tag_configure('time', foreground=MUTED, font=(self.host.FONT_MONO, 11))
        self.transcript.tag_configure('now', background='#30211e')
        for i, color in enumerate(SPEAKER_COLORS):
            self.transcript.tag_configure(f'who{i}', foreground=color, font=(self.host.FONT_UI, 12, 'bold'))
        self.transcript.bind('<Button-1>', self._click_line)
        self.transcript.configure(state='disabled')
        ctk.CTkLabel(view, text='Clique numa linha para ouvir dali. Renomeie quem falou nos campos acima.',
                     text_color=MUTED, font=(self.host.FONT_UI, 10), anchor='w').pack(fill='x', padx=16, pady=(0, 8))
        return view

    def _show(self, view):
        for v in (self.ready_view, self.work_view, self.done_view):
            if v is not view:
                v.pack_forget()
        view.pack(fill='both', expand=True)
        if view is self.ready_view:
            self._refresh_previous()

    # -- ciclo da aba --------------------------------------------------------
    def shown(self):
        """A aba abriu: liga os medidores (abrir o mic so com alguem olhando)."""
        self.visible = True
        if self.sources is None:
            self.sources = (rec.Source(rec.MIC), rec.Source(rec.PC))
        self._set_ready_text()

    def hidden(self):
        self.visible = False
        if not self.recording and self.sources is not None:
            for source in self.sources:
                source.close()
            self.sources = None

    def close(self):
        self.closed = True
        if self.recording:
            self._stop_capture()  # o bruto fica no cache: a proxima abertura oferece salvar
        if self.sources is not None:
            for source in self.sources:
                source.close()
        self.player.stop()
        if self.abort is not None:
            self.abort.set()

    # -- gravacao ------------------------------------------------------------
    def start(self):
        if self.recording or self.abort is not None:
            return
        if rec.pending_recording() is not None:
            self.ready_status.configure(text='Há uma gravação interrompida: salve ou descarte antes de começar outra.')
            return
        if self.sources is None:
            self.sources = (rec.Source(rec.MIC), rec.Source(rec.PC))
        mic, pc, _ = rec.raw_paths()
        self.started = datetime.now()
        self._write_state()
        self.sources[0].start_recording(mic)
        self.sources[1].start_recording(pc)
        self.recording, self.paused = True, False
        self.elapsed, self.resumed_at = 0.0, time.monotonic()
        self.start_btn.pack_forget()
        self.pause_btn.configure(text='Pausar')
        self.pause_btn.pack(side='left')
        self.stop_btn.pack(side='left', padx=(8, 0))
        self.import_btn.configure(state='disabled')
        self.previous.configure(state='disabled')
        self._set_ready_text()

    def _write_state(self):
        """Nome e idioma valem no momento em que para: dao pra mudar durante a call."""
        title = self.title_entry.get().strip() or f'Reunião {self.started:%H:%M}'
        rec.raw_paths()[2].write_text(json.dumps({
            'title': title, 'started_at': self.started.isoformat(timespec='seconds'),
            'language': self.language.get(), 'voices': self.voices.get()}, ensure_ascii=False), encoding='utf-8')

    def pause(self):
        if not self.recording:
            return
        self.paused = not self.paused
        for source in self.sources:
            source.set_paused(self.paused)
        if self.paused:
            self.elapsed += time.monotonic() - self.resumed_at
        else:
            self.resumed_at = time.monotonic()
        self.pause_btn.configure(text='Retomar' if self.paused else 'Pausar')
        self._set_ready_text()

    def _stop_capture(self):
        for source in self.sources:
            source.stop_recording()
        self.recording = False
        self.pause_btn.pack_forget()
        self.stop_btn.pack_forget()
        self.start_btn.pack(side='left')
        self.import_btn.configure(state='normal')
        self.previous.configure(state='readonly')
        if not self.visible:
            self.hidden()

    def stop(self):
        if not self.recording:
            return
        self._write_state()
        self._stop_capture()
        self._save_pending()

    def _save_pending(self):
        info = rec.pending_recording()
        if info is None:
            self._set_ready_text('Nada foi gravado.')
            return
        started = datetime.fromisoformat(info['started_at']) if info.get('started_at') else datetime.now()
        title = info.get('title') or f'Reunião {started:%H:%M}'

        def work():
            # A pasta so nasce aqui: se o modelo nao estiver pronto, o bruto segue no cache.
            folder = rec.new_meeting_dir(title, started)
            rec.write_manifest(folder, {'app': 'sussurro', 'version': 1, 'title': title,
                                        'started_at': started.isoformat(timespec='seconds'),
                                        'duration_secs': info['duration_secs'],
                                        'language': info.get('language', 'pt'), 'imported': None,
                                        'speakers': None, 'voices': info.get('voices', True),
                                        'names': {}, 'lines': []})
            mic, pc, _ = rec.raw_paths()
            rec.export(mic, pc, folder)
            rec.discard_pending()  # os Opus da pasta ja guardam tudo
            return self._transcribe_folder(folder)

        if not self._run_job('Salvando o áudio...', work):
            self._offer_recovery()

    # -- transcricao ---------------------------------------------------------
    def _run_job(self, stage, work):
        """Roda `work` numa thread, com o modelo do ditado emprestado (nao troca no meio)."""
        transcriber = self.host.transcriber
        try:
            transcriber.begin_meeting_job()
        except RuntimeError as error:
            self._set_ready_text(f'{error} O áudio está guardado; tente de novo em seguida.')
            return False
        self.abort = threading.Event()
        self._show(self.work_view)
        self.work_stage.configure(text=stage)
        self.work_bar.set(0)
        self.work_lines.configure(state='normal')
        self.work_lines.delete('1.0', 'end')
        self.work_lines.configure(state='disabled')

        def run():
            try:
                self.events.put(('done', work()))
            except meeting.Cancelled:
                self.events.put(('cancelled', None))
            except Exception as error:  # a gravacao fica na pasta; o erro vai pra tela
                self.events.put(('error', f'{type(error).__name__}: {error}'))
            finally:
                transcriber.end_meeting_job()

        threading.Thread(target=run, name='sussurro-reuniao', daemon=True).start()
        return True

    def _report(self, kind, value):
        self.events.put((kind, value))

    def _transcribe_folder(self, folder):
        """Transcreve a pasta (gravacao com duas trilhas ou arquivo importado) e grava o resultado."""
        manifest = rec.read_manifest(folder)
        transcriber = self.host.transcriber
        model, lock = transcriber.model, transcriber._model_lock
        language = manifest.get('language', 'pt')
        started = time.perf_counter()
        if manifest.get('imported'):
            track = meeting.load_track(folder / 'audio.ogg')
            transcript = meeting.transcribe_single(track, language, manifest.get('speakers'), model,
                                                   report=self._report, abort=self.abort, lock=lock,
                                                   hotwords=transcriber.library.hotwords)
        else:
            mic = meeting.load_track(folder / rec.TRACKS / 'mic.ogg')
            pc = meeting.load_track(folder / rec.TRACKS / 'pc.ogg')
            transcript = meeting.transcribe_meeting(mic, pc, language, model, find_voices=manifest.get('voices', True),
                                                    report=self._report, abort=self.abort, lock=lock,
                                                    hotwords=transcriber.library.hotwords)
        for line in transcript.lines:
            line.text, _ = transcriber.library.apply(line.text)  # Biblioteca tambem vale aqui
        config = transcriber.model_config
        manifest.update(duration_secs=transcript.duration_secs, language=language,
                        detected_language=transcript.language, model=config.label if config else None,
                        transcribe_s=round(time.perf_counter() - started, 1),
                        lines=[vars(line) for line in transcript.lines])
        rec.write_manifest(folder, manifest)
        self._write_markdown(folder, manifest)
        return folder

    def _write_markdown(self, folder, manifest):
        transcript = meeting.Transcript([meeting.Line(**line) for line in manifest['lines']],
                                        manifest.get('detected_language') or manifest.get('language', 'pt'),
                                        manifest.get('duration_secs', 0))
        started = datetime.fromisoformat(manifest['started_at'])
        text = meeting.to_markdown(manifest['title'], f'{started:%d/%m/%Y %H:%M}', transcript,
                                   manifest.get('names') or {})
        (Path(folder) / rec.TRANSCRIPT).write_text(text, encoding='utf-8')

    def _cancel_job(self):
        if self.abort is not None:
            self.abort.set()
            self.work_stage.configure(text='Cancelando... o áudio fica salvo na pasta.')

    def _again(self):
        if self.folder is None or self.abort is not None:
            return
        self.player.stop()
        folder = self.folder
        self.manifest['language'] = self.again_language.get()
        rec.write_manifest(folder, self.manifest)
        self._run_job('Transcrevendo de novo...', lambda: self._transcribe_folder(folder))

    def import_file(self):
        if self.recording or self.abort is not None:
            return
        path = filedialog.askopenfilename(parent=self, title='Importar áudio para transcrever',
                                          filetypes=[('Áudio', '*.mp3 *.m4a *.ogg *.opus *.wav *.flac *.webm *.mp4 *.aac'),
                                                     ('Todos', '*')])
        if not path:
            return
        source = Path(path)
        started = datetime.fromtimestamp(source.stat().st_mtime)
        title = self.title_entry.get().strip() or source.stem
        speakers = self.speakers.get()
        language = self.language.get()

        def work():
            folder = rec.new_meeting_dir(title, started)
            rec.write_manifest(folder, {'app': 'sussurro', 'version': 1, 'title': title,
                                        'started_at': started.isoformat(timespec='seconds'), 'duration_secs': 0,
                                        'language': language, 'imported': source.name,
                                        'speakers': None if speakers == 'Automático' else int(speakers),
                                        'names': {}, 'lines': []})
            rec.import_audio(source, folder)
            return self._transcribe_folder(folder)
        self._run_job(f'Importando {source.name}...', work)

    # -- recuperacao ---------------------------------------------------------
    def _offer_recovery(self):
        info = rec.pending_recording()
        if info is None or self.recording:
            self.recovery.pack_forget()
            return
        minutes, seconds = divmod(info['duration_secs'], 60)
        self.recovery_label.configure(text=f'Gravação interrompida encontrada: {info.get("title", "sem nome")} '
                                           f'({minutes:02}:{seconds:02}).')
        self.recovery.pack(fill='x', padx=16, pady=(12, 0), after=self.ready_view.winfo_children()[0])

    def _save_recovery(self):
        self.recovery.pack_forget()
        self._save_pending()

    def _discard_recovery(self):
        rec.discard_pending()
        self.recovery.pack_forget()
        self._set_ready_text('Gravação interrompida descartada.')

    # -- tela de pronta ------------------------------------------------------
    def open_meeting(self, folder):
        self.player.stop()
        self.folder = Path(folder)
        self.manifest = rec.read_manifest(self.folder)
        m = self.manifest
        self.done_title.set(m['title'])
        self.again_language.set(m.get('language', 'pt'))
        started = datetime.fromisoformat(m['started_at'])
        language = meeting.LANGUAGES.get(m.get('detected_language') or m.get('language'), m.get('language'))
        extra = f' · {m["model"]}' if m.get('model') else ''
        took = f' · transcrito em {m["transcribe_s"]:.0f} s' if m.get('transcribe_s') else ''
        self.done_meta.configure(text=f'{started:%d/%m/%Y %H:%M} · {meeting.clock(m.get("duration_secs", 0) * 1000)}'
                                      f' · {language}{extra}{took}')
        for child in self.names_frame.winfo_children():
            child.destroy()
        self.name_vars = {}
        labels = list(dict.fromkeys(line['speaker'] for line in m['lines']))
        names = m.get('names') or {}
        for i, label in enumerate(labels):
            var = tk.StringVar(value=names.get(label, label))
            entry = ctk.CTkEntry(self.names_frame, textvariable=var, width=130, height=28, font=self.font,
                                 fg_color=FIELD, border_color=SPEAKER_COLORS[i % len(SPEAKER_COLORS)],
                                 text_color=INK)
            entry.grid(row=i // 5, column=i % 5, padx=(0, 6), pady=(0, 4), sticky='w')
            entry.bind('<Return>', lambda _e: self._apply_names())
            entry.bind('<FocusOut>', lambda _e: self._apply_names())
            _select_on_focus(entry)
            self.name_vars[label] = var
        self._render_transcript()
        self._show(self.done_view)

    def _render_transcript(self):
        m = self.manifest
        names = m.get('names') or {}
        order = {label: i for i, label in enumerate(self.name_vars)}
        text = self.transcript
        text.configure(state='normal')
        text.delete('1.0', 'end')
        self.line_at = []
        if not m['lines']:
            text.insert('end', 'Nenhuma fala reconhecida.')
        for line in m['lines']:
            row = int(text.index('end-1c').split('.')[0])
            self.line_at.append((row, line['start_ms']))
            text.insert('end', meeting.clock(line['start_ms']) + '  ', 'time')
            who = order.get(line['speaker'], 0) % len(SPEAKER_COLORS)
            text.insert('end', names.get(line['speaker'], line['speaker']) + ': ', f'who{who}')
            text.insert('end', line['text'] + '\n')
        text.configure(state='disabled')

    def _apply_names(self):
        if self.manifest is None:
            return
        names = {label: var.get().strip() for label, var in self.name_vars.items()
                 if var.get().strip() and var.get().strip() != label}
        if names == (self.manifest.get('names') or {}):
            return
        self.manifest['names'] = names
        rec.write_manifest(self.folder, self.manifest)
        self._write_markdown(self.folder, self.manifest)
        self._render_transcript()

    def _rename(self):
        title = self.done_title.get().strip()
        if self.manifest is None or not title or title == self.manifest['title']:
            return
        self.player.stop()
        self.manifest['title'] = title
        self.folder = rec.rename_meeting(self.folder, title)
        rec.write_manifest(self.folder, self.manifest)
        self._write_markdown(self.folder, self.manifest)

    def _copy(self):
        text = (self.folder / rec.TRANSCRIPT).read_text(encoding='utf-8')
        self.clipboard_clear()
        self.clipboard_append(text)
        meta = self.done_meta.cget('text')
        self.done_meta.configure(text='Transcrição copiada.')
        self.after(1600, lambda: self.done_meta.configure(text=meta))

    def _open_folder(self):
        subprocess.Popen(['xdg-open', str(self.folder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _toggle_play(self):
        if self.player.playing():
            self.player.stop()
        else:
            self._play_from(0)

    def _play_from(self, ms):
        audio = self.folder / 'audio.ogg'
        if audio.is_file():
            self.player.play(audio, ms)

    def _click_line(self, event):
        row = int(self.transcript.index(f'@{event.x},{event.y}').split('.')[0])
        for line_row, start_ms in reversed(self.line_at):
            if line_row <= row:
                self._play_from(start_ms)
                return

    def _new(self):
        self.player.stop()
        self.folder = self.manifest = None
        self.title_entry.delete(0, 'end')
        self._show(self.ready_view)
        self._set_ready_text()

    def _refresh_previous(self):
        try:
            meetings = rec.list_meetings()[:30]
        except OSError:
            meetings = []
        self._previous = {f'{folder.name[:8]} · {m.get("title", folder.name)}': folder for folder, m in meetings}
        self.previous.configure(values=list(self._previous) or ['—'])
        self.previous.set('—')

    def _open_previous(self, choice):
        folder = getattr(self, '_previous', {}).get(choice)
        if folder is not None and not self.recording:
            self.open_meeting(folder)

    # -- laco da interface ---------------------------------------------------
    def _set_ready_text(self, message=None):
        if self.recording:
            state = 'Pausado: nada é gravado até retomar.' if self.paused else 'Gravando os dois lados.'
        else:
            state = 'Pronto. Confira se os dois medidores mexem antes de começar.'
        self.state_label.configure(text=state)
        if message is not None:
            self.ready_status.configure(text=message)

    def _draw_meters(self):
        for (canvas, color), source in zip(self.canvases, self.sources or (None, None)):
            canvas.delete('all')
            width, height = max(canvas.winfo_width(), 10), max(canvas.winfo_height(), 10)
            mid = height / 2
            canvas.create_line(0, mid, width, mid, fill='#3a3b40')
            if source is None:
                continue
            levels = source.levels()
            step = width / len(levels)
            fill = MUTED if (self.recording and self.paused) else color
            for i, peak in enumerate(levels):
                h = rec.to_meter(peak) * (mid - 3)
                if h > 0.5:
                    x = i * step
                    canvas.create_rectangle(x, mid - h, x + max(step - 1, 1), mid + h, fill=fill, width=0)

    def _tick(self):
        if self.closed:
            return
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            self._handle(kind, value)
        if self.recording or self.visible:
            seconds = self.elapsed + (time.monotonic() - self.resumed_at if self.recording and not self.paused else 0)
            if self.recording:
                self.clock.configure(text=meeting.clock(seconds * 1000))
            if self.visible:
                self._draw_meters()
        playing = self.folder is not None and self.player.playing()
        if playing:
            self._highlight(self.player.position_ms())
        self.play_btn.configure(text='Parar' if playing else 'Ouvir')
        self.after(60, self._tick)

    def _highlight(self, ms):
        self.transcript.tag_remove('now', '1.0', 'end')
        current = None
        for row, start in self.line_at:
            if start <= ms:
                current = row
        if current is not None:
            self.transcript.tag_add('now', f'{current}.0', f'{current}.end')

    def _handle(self, kind, value):
        if kind == 'stage':
            self.work_stage.configure(text=value)
        elif kind == 'progress':
            self.work_bar.set(max(0.0, min(1.0, value)))
        elif kind == 'line':
            self.work_lines.configure(state='normal')
            self.work_lines.insert('end', value + '\n')
            self.work_lines.see('end')
            self.work_lines.configure(state='disabled')
        elif kind == 'done':
            self.abort = None
            self.open_meeting(value)
        elif kind in ('cancelled', 'error'):
            self.abort = None
            self._show(self.ready_view)
            self._offer_recovery()
            self._set_ready_text('Transcrição cancelada; o áudio ficou na pasta da reunião.' if kind == 'cancelled'
                                 else f'ERRO na reunião: {value}. O áudio ficou salvo.')

    # -- atalhos (IPC) -------------------------------------------------------
    def command(self, verb):
        """`sussurro meeting-start|meeting-stop|meeting-pause`: para atalhos do Hyprland."""
        if verb == 'start':
            self.start()
        elif verb == 'stop':
            self.stop()
        elif verb == 'pause':
            self.pause()
