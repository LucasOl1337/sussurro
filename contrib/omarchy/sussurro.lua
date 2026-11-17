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
-- nao sombreia e nao toma foco ao abrir ou passar o mouse. no_focus impediria os cliques.
o.window({ class = "^SussurroBar$" }, {
  float = true,
  pin = true,
  no_follow_mouse = true,
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

-- Indicador de reuniao (classe SussurroMeeting, 112x26): pilula discreta com tempo,
-- pausar e esconder. Mesmas regras da barra; fica onde foi solta, em todo workspace.
o.window({ class = "^SussurroMeeting$" }, {
  float = true,
  pin = true,
  no_follow_mouse = true,
  no_initial_focus = true,
  no_anim = true,
  no_shadow = true,
  no_blur = true,
  no_dim = true,
  border_size = 0,
  rounding = 13,
  size = { 112, 26 },
  tag = "-default-opacity",
  opacity = "1 1",
  render_unfocused = true,
})

-- Sem barra de titulo: em workspace flutuante o Omarchy 4.5 poe a barra do hyprbars em
-- toda janela, e na capsula ela cobria os botoes e o arraste. O campo hyprbars:no_bar so
-- existe depois que o Omarchy carrega o plugin, o que acontece depois deste arquivo
-- (toggles.lua) ou so quando um workspace vira flutuante: tenta agora e de novo a cada
-- vez que a barra ou o indicador aparecem. Sem o plugin, nao ha barra a tirar, e pedir
-- o campo assim mesmo vira erro em `hyprctl configerrors` (o pcall nao segura o aviso).
local sussurro_sem_titulo = false
local function sussurro_hyprbars()
  for _, plugin in ipairs(hl.get_loaded_plugins and hl.get_loaded_plugins() or {}) do
    if plugin.name == "hyprbars" then
      return true
    end
  end
  return false
end
local function sussurro_tira_titulo()
  if not sussurro_sem_titulo and sussurro_hyprbars() then
    sussurro_sem_titulo = pcall(hl.window_rule, {
      name = "sussurro-sem-titulo",
      match = { class = "^Sussurro(Bar|Meeting)$" },
      ["hyprbars:no_bar"] = true,
    })
  end
end
sussurro_tira_titulo()
if hl.on then
  hl.on("window.open", function(window)
    local class = window and window.class
    if class == "SussurroBar" or class == "SussurroMeeting" then
      sussurro_tira_titulo()
    end
  end)
end
