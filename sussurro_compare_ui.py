"""Comparison tab: one recording reused across model/device/run choices."""
import queue
import threading
import time
import wave
from tkinter import filedialog

import customtkinter as ctk
import numpy as np

from sussurro_compare import ClipRecorder, Comparison, SAMPLE_RATE, MAX_SECONDS
from sussurro_models import MODEL_LABELS

BG = '#1b1c20'
FIELD = '#25262b'
INK = '#f2f3f3'
MUTED = '#9ba0a4'
ACCENT = '#f0500a'


class ComparisonPanel(ctk.CTkFrame):
    def __init__(self, parent, host, resampler_factory):
        super().__init__(parent, fg_color='transparent')
        self.host = host
        self.resampler_factory = resampler_factory
        self.clip = None
        self.recorder = None
        self.runner = None
        self.reserved = False
        self.closed = False
        self.events = queue.Queue()
        self.rows = {}
        self.model_vars = {}
        self.font = (host.FONT_UI, 12)
        ctk.CTkLabel(self, text='Uma frase. Todos os modelos.', font=(host.FONT_DISPLAY, 23),
                     text_color=INK, anchor='w').pack(fill='x', padx=16, pady=(12, 0))
        ctk.CTkLabel(self, text='Grave ate 30 s no microfone selecionado. Todos recebem exatamente o mesmo audio.',
                     text_color=MUTED, font=self.font, anchor='w').pack(fill='x', padx=16)
        choices = ctk.CTkFrame(self, fg_color='transparent')
        choices.pack(fill='x', padx=16, pady=8)
        for i, model in enumerate(m for m in MODEL_LABELS if m != 'auto'):
            var = ctk.BooleanVar(value=True)
            box = ctk.CTkCheckBox(choices, text=model.replace('large-v3-turbo', 'Turbo').replace('parakeet-tdt-0.6b-v3', 'Parakeet'),
                                 variable=var, width=140, font=self.font,
                                 fg_color=ACCENT, hover_color='#d64708')
            box.grid(row=i // 3, column=i % 3, sticky='w', pady=4, padx=(0, 14))
            self.model_vars[model] = (var, box)
        options = ctk.CTkFrame(self, fg_color='transparent')
        options.pack(fill='x', padx=16)
        self.device = ctk.CTkComboBox(options, values=['Mesmo do ditado', 'CPU', 'GPU NVIDIA'],
                                      state='readonly', width=170, font=self.font)
        self.device.set('Mesmo do ditado')
        self.device.pack(side='left')
        self.mode = ctk.CTkComboBox(options, values=['Paralelo (2 por vez)', 'Individual (1 por vez)'],
                                    state='readonly', width=200, font=self.font)
        self.mode.set('Paralelo (2 por vez)')
        self.mode.pack(side='left', padx=8)
        self.language = ctk.CTkComboBox(options, values=['pt', 'en', 'auto'], state='readonly',
                                        width=80, font=self.font)
        self.language.set(host.settings['language'])
        self.language.pack(side='left')
        buttons = ctk.CTkFrame(self, fg_color='transparent')
        buttons.pack(fill='x', padx=16, pady=(12, 6))
        self.record = ctk.CTkButton(buttons, text='Gravar frase', command=self.record_toggle,
                                    fg_color=ACCENT, hover_color='#d64708', text_color='#16181a',
                                    width=170, height=34, font=self.font)
        self.record.pack(side='left')
        self.again = ctk.CTkButton(buttons, text='Comparar novamente', command=self.compare,
                                   width=164, state='disabled', font=self.font)
        self.again.pack(side='left', padx=8)
        self.open = ctk.CTkButton(buttons, text='Carregar WAV', command=self.open_wav,
                                  width=120, font=self.font)
        self.open.pack(side='left')
        self.cancel = ctk.CTkButton(buttons, text='Cancelar', command=self.cancel_run,
                                    width=94, state='disabled', font=self.font)
        self.cancel.pack(side='left', padx=8)
        self.status = ctk.CTkLabel(self, text='Escolha os modelos e grave uma frase.',
                                    font=self.font, text_color=INK, anchor='w', justify='left', wraplength=670)
        self.status.pack(fill='x', padx=16, pady=(0, 4))
        ctk.CTkLabel(self, text='Transcricao = inferencia apos aquecer. Total = fila + preparo + carga + transcricao.\n'
                               'Paralelo disputa recursos; Individual facilita comparar velocidade. Sem correcoes da Biblioteca.\n'
                               'Primeiro uso baixa os pesos. Modelos de teste sao liberados ao terminar ou cancelar.',
                     font=(host.FONT_UI, 10), text_color=MUTED, anchor='w', justify='left').pack(fill='x', padx=16, pady=(0, 8))
        self.results = ctk.CTkScrollableFrame(self, fg_color=BG)
        self.results.pack(fill='both', expand=True, padx=12, pady=(0, 12))
        for button in (self.again, self.open, self.cancel):
            button.configure(fg_color=FIELD, hover_color='#30211e', text_color=INK,
                             border_width=1, border_color='#454750')
        for box in (self.device, self.mode, self.language):
            box.configure(fg_color=FIELD, border_color='#2f3036', button_color=FIELD,
                          button_hover_color='#30211e', dropdown_fg_color=FIELD,
                          dropdown_hover_color='#30211e', dropdown_text_color=INK, text_color=INK)
        self.after(80, self.poll)

    def acquire(self):
        if self.reserved:
            return True
        try:
            self.host.transcriber.acquire_comparison()
        except RuntimeError as error:
            self.status.configure(text=str(error))
            return False
        self.reserved = True
        self.host.record_btn.configure(state='disabled')
        self.host.apply_model_btn.configure(state='disabled')
        return True

    def release(self):
        if self.reserved:
            self.reserved = False
            self.host.transcriber.comparing.clear()
            if not self.closed:
                ready = self.host.transcriber.model is not None and not self.host.transcriber.model_loading.is_set()
                self.host.record_btn.configure(state='normal' if ready else 'disabled')
                self.host.apply_model_btn.configure(state='normal')

    def controls(self, busy, recording=False):
        self.record.configure(state='normal' if recording or not busy else 'disabled',
                              text='Parar e comparar' if recording else 'Gravar frase')
        self.again.configure(state='normal' if self.clip is not None and not busy else 'disabled')
        self.open.configure(state='disabled' if busy else 'normal')
        self.cancel.configure(state='normal' if busy else 'disabled')
        for _, box in self.model_vars.values():
            box.configure(state='disabled' if busy else 'normal')
        for box in (self.mode, self.device, self.language):
            box.configure(state='disabled' if busy else 'readonly')

    def record_toggle(self):
        if self.recorder is not None:
            recorder, self.recorder = self.recorder, None
            try:
                clip = recorder.stop()
                if len(clip) < SAMPLE_RATE // 4:
                    raise ValueError('Gravacao curta demais. Fale a frase antes de parar.')
                self.clip = clip
            except Exception as error:
                self.status.configure(text=f'Falha na gravacao: {error}')
                self.release()
                self.controls(False)
                return
            self.compare()
            return
        if not any(var.get() for var, _ in self.model_vars.values()):
            self.status.configure(text='Selecione pelo menos um modelo.')
            return
        if not self.acquire():
            return
        recorder = ClipRecorder(self.host._device_index(), self.resampler_factory)
        try:
            recorder.start()
        except Exception as error:
            self.status.configure(text=f'Nao foi possivel abrir o microfone: {error}')
            self.release()
            return
        self.recorder = recorder
        self.recorded_at = time.perf_counter()
        self.controls(True, recording=True)

    def open_wav(self):
        if self.reserved:
            return
        path = filedialog.askopenfilename(parent=self, title='Comparar uma frase em WAV',
                                         filetypes=[('Audio WAV', '*.wav')])
        if not path:
            return
        try:
            with wave.open(path) as wav:
                rate, channels = wav.getframerate(), wav.getnchannels()
                if wav.getsampwidth() != 2 or channels not in (1, 2):
                    raise ValueError('Use WAV PCM de 16 bits, mono ou stereo.')
                if not 0 < wav.getnframes() <= rate * MAX_SECONDS:
                    raise ValueError('Escolha um audio com ate 30 segundos.')
                audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768
                audio = audio.reshape(-1, channels).mean(axis=1)
            self.clip = self.resampler_factory(rate).process(audio) if rate != SAMPLE_RATE else audio
            self.status.configure(text=f'Audio carregado: {len(self.clip) / SAMPLE_RATE:.1f} s. Clique em Comparar novamente.')
            self.controls(False)
        except Exception as error:
            self.status.configure(text=f'Nao foi possivel carregar o WAV: {error}')

    def compare(self):
        models = [model for model, (var, _) in self.model_vars.items() if var.get()]
        if self.clip is None or not len(self.clip) or not models:
            self.status.configure(text='Grave uma frase e selecione pelo menos um modelo.')
            self.release()
            self.controls(False)
            return
        if not self.acquire():
            return
        device = self.device.get()
        if device == 'Mesmo do ditado':
            config = self.host.transcriber.model_config
            device = config.device if config else 'cpu'
        else:
            device = 'cpu' if device == 'CPU' else 'cuda'
        for child in self.results.winfo_children():
            child.destroy()
        self.rows = {}
        for model in models:
            row = ctk.CTkFrame(self.results, fg_color=FIELD)
            row.pack(fill='x', pady=(0, 8))
            label = ctk.CTkLabel(row, text=f'{model} · Na fila', text_color=INK,
                                 font=(self.host.FONT_UI, 13, 'bold'), anchor='w')
            label.pack(fill='x', padx=10, pady=(6, 0))
            metrics = ctk.CTkLabel(row, text='—', font=(self.host.FONT_MONO, 11),
                                   text_color=MUTED, anchor='w', justify='left')
            metrics.pack(fill='x', padx=10)
            text = ctk.CTkTextbox(row, height=65, font=(self.host.FONT_UI, 13),
                                  fg_color='transparent', text_color=INK, wrap='word', state='disabled')
            text.pack(fill='x', padx=6, pady=(0, 6))
            self.rows[model] = (label, metrics, text)
        self.results._parent_canvas.yview_moveto(0)
        self.controls(True)
        self.status.configure(text=f'Mesma frase de {len(self.clip) / SAMPLE_RATE:.1f} s · {device.upper()} · {self.mode.get()}')
        self.runner = Comparison(self.events.put)
        threading.Thread(target=self.runner.run,
                         args=(self.clip.copy(), models, device, self.language.get(), self.mode.get().startswith('Paralelo')),
                         daemon=True).start()

    def handle_event(self, event):
        kind = event['event']
        if kind == 'finished':
            self.runner = None
            self.release()
            self.controls(False)
            self.status.configure(text=event.get('error') or ('Comparacao cancelada.' if event['cancelled'] else
                                  'Comparacao concluida. Voce pode repetir com o mesmo audio.'))
        elif kind == 'error':
            self.status.configure(text=event['error'])
        elif event.get('model') in self.rows:
            model = event['model']
            label, metrics, text = self.rows[model]
            if kind == 'stage':
                label.configure(text=f'{model} · {event["stage"]}')
            elif kind == 'cancelled':
                label.configure(text=f'{model} · Cancelado')
            elif kind == 'result':
                error = event.get('error')
                if error and ('out of memory' in error.lower() or 'alloc_failed' in error.lower()):
                    error = 'Memoria insuficiente. Tente Individual, CPU ou libere memoria de outros apps.\n' + error
                label.configure(text=f'{model} · ' + ('Falhou' if error else event['device'].upper() + ' · ' + event['compute_type']))
                if not error:
                    metrics.configure(text=f'Transcricao: {event["infer_s"]:.3f} s · Carga: {event["load_s"]:.2f} s · Total: {event["elapsed_s"]:.2f} s\n'
                                           f'Preparo/download: {event["prep_s"]:.2f} s · Audio: {event["audio_s"]:.1f} s')
                text.configure(state='normal')
                text.delete('1.0', 'end')
                text.insert('1.0', error or event['text'] or '(Nenhuma fala reconhecida.)')
                text.configure(state='disabled')

    def poll(self):
        if self.closed:
            return
        if self.recorder:
            self.status.configure(text=f'Gravando: {min(time.perf_counter() - self.recorded_at, MAX_SECONDS):.1f} / {MAX_SECONDS} s. Fale e clique em Parar e comparar.')
            if self.recorder.full.is_set():
                self.record_toggle()
        while True:
            try:
                self.handle_event(self.events.get_nowait())
            except queue.Empty:
                break
        self.after(80, self.poll)

    def cancel_run(self):
        if self.recorder:
            recorder, self.recorder = self.recorder, None
            try:
                recorder.stop()
            except Exception:
                pass
            self.release()
            if not self.closed:
                self.controls(False)
                self.status.configure(text='Gravacao cancelada.')
        if self.runner:
            self.runner.cancel()
            if not self.closed:
                self.cancel.configure(state='disabled')
                self.status.configure(text='Cancelando e liberando os modelos...')

    def close(self):
        self.closed = True
        self.cancel_run()
