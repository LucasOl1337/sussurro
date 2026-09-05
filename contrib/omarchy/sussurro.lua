-- Keep the dictation UI alive on other workspaces so CUDA/Tk do not freeze
-- when Sussurro is not the focused window.
o.window({ class = "^Tk$", title = "^Sussurro$" }, {
  render_unfocused = true,
  tag = "-default-opacity",
  opacity = "1 1",
})
o.window({ class = "^Sussurro$" }, {
  render_unfocused = true,
  tag = "-default-opacity",
  opacity = "1 1",
})

-- Barra de gravacao do Sussurro (janela Tk/XWayland de 152x40, classe SussurroBar).
-- Gerenciada como flutuante fixa: o Hyprland recorta os cantos (rounding), nao anima,
-- nao sombreia e nunca recebe foco, entao o Enter/colar continua indo pro app de baixo.
o.window({ class = "^SussurroBar$" }, {
  float = true,
  pin = true,
  no_focus = true,
  no_initial_focus = true,
  no_anim = true,
  no_shadow = true,
  no_blur = true,
  no_dim = true,
  border_size = 0,
  rounding = 20,
  size = { 152, 40 },
  tag = "-default-opacity",
  opacity = "1 1",
  render_unfocused = true,
})
