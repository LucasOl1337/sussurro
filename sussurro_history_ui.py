"""Escolhas de uma nova transcricao, sem alterar as preferencias do ditado."""
import customtkinter as ctk

from sussurro_models import MODEL_LABELS, PARAKEET


class RetranscribeDialog(ctk.CTkToplevel):
    def __init__(self, host, entry, on_submit):
        super().__init__(host.root)
        self.title("Refazer transcrição")
        self.configure(fg_color="#1b1c20")
        self.resizable(False, False)
        self.transient(host.root)
        self.on_submit = on_submit
        self.model_labels = {
            key: ("Whisper · " + label if key not in ("auto", PARAKEET) else label)
            for key, label in MODEL_LABELS.items()
        }
        self.device_labels = host.device_labels
        font = (host.FONT_UI, 12)

        ctk.CTkLabel(self, text="Refazer transcrição", font=(host.FONT_UI, 21, "bold"),
                     text_color="#f2f3f3", anchor="w").pack(fill="x", padx=24, pady=(20, 6))
        ctk.CTkLabel(self, text="O áudio salvo será transcrito de novo. Estas escolhas valem só\n"
                               "pra esta tentativa; seu modelo de ditado continua o mesmo.",
                     font=font, text_color="#c3c6c8", justify="left", anchor="w").pack(
                         fill="x", padx=24, pady=(0, 12))
        preview = ctk.CTkTextbox(self, width=520, height=100, font=font,
                                fg_color="#25262b", text_color="#c3c6c8", wrap="word")
        preview.pack(fill="x", padx=24, pady=(0, 10))
        preview.insert("1.0", entry.get("text") or "Sem transcrição. O áudio foi preservado.")
        preview.configure(state="disabled")

        def choice(label, values):
            ctk.CTkLabel(self, text=label, font=(host.FONT_UI, 10, "bold"),
                         text_color="#9ba0a4", anchor="w").pack(fill="x", padx=24)
            box = ctk.CTkComboBox(self, values=list(values), state="readonly", font=font,
                                  height=34, fg_color="#25262b", border_color="#2f3036",
                                  button_color="#2f3036", button_hover_color="#454750",
                                  dropdown_fg_color="#25262b", dropdown_hover_color="#454750",
                                  dropdown_text_color="#f2f3f3", text_color="#f2f3f3")
            box.pack(fill="x", padx=24, pady=(0, 8))
            return box

        config = host.transcriber.model_config
        self.model = choice("MOTOR / MODELO", self.model_labels.values())
        self.model.set(self.model_labels.get(config.model if config else "auto", MODEL_LABELS["auto"]))
        self.device = choice("EXECUTAR EM", self.device_labels.values())
        self.device.set(self.device_labels[config.device if config else "auto"])
        self.language = choice("IDIOMA", ["pt", "en", "auto"])
        self.language.set(entry.get("language", host.transcriber.language))

        ctk.CTkLabel(self, text="O novo texto substitui o atual no histórico. Se falhar, o atual fica salvo.\n"
                               "O primeiro uso de um modelo pode precisar de download.\n"
                               "Parakeet usa GPU NVIDIA e detecta o idioma automaticamente.",
                     font=(host.FONT_UI, 11), text_color="#9ba0a4", justify="left", anchor="w").pack(
                         fill="x", padx=24, pady=(4, 0))
        self.error = ctk.CTkLabel(self, text=entry.get("retry_error", ""), wraplength=510,
                                  font=font, text_color="#f07944", justify="left", anchor="w")
        self.error.pack(fill="x", padx=24, pady=(4, 0))
        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.pack(fill="x", padx=24, pady=(8, 20))
        ctk.CTkButton(buttons, text="Cancelar", command=self.destroy, width=110, height=34,
                      fg_color="#25262b", hover_color="#454750", text_color="#f2f3f3",
                      font=font).pack(side="left")
        self.submit = ctk.CTkButton(buttons, text="Refazer transcrição", command=self._submit,
                                     width=180, height=34, fg_color="#f0500a",
                                     hover_color="#d64708", text_color="#16181a", font=font)
        self.submit.pack(side="right")
        self.bind("<Escape>", lambda _event: self.destroy())
        self.after(100, self._focus)

    def _focus(self):
        self.grab_set()
        self.focus_set()

    def _submit(self):
        selection = {
            "whisper_model": next(k for k, v in self.model_labels.items() if v == self.model.get()),
            "whisper_device": next(k for k, v in self.device_labels.items() if v == self.device.get()),
        }
        if selection["whisper_model"] == PARAKEET and selection["whisper_device"] == "cpu":
            self.error.configure(text="Parakeet precisa de GPU NVIDIA. Escolha GPU ou Automático.")
            return
        try:
            self.on_submit(selection, self.language.get())
        except Exception as error:
            self.error.configure(text=str(error))
            return
        self.destroy()
