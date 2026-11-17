"""A barra precisa receber ponteiro, sem tomar foco ao abrir ou passar o mouse."""
import shutil
import subprocess
import unittest
from pathlib import Path

import app


class RecorderBarRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lua = shutil.which('lua')
        if not lua:
            raise unittest.SkipTest('lua indisponivel para avaliar as regras do Omarchy')
        rule = Path(__file__).resolve().parents[1] / 'contrib/omarchy/sussurro.lua'
        # Executa o arquivo entregue, capturando so a regra da barra, sem Hyprland.
        result = subprocess.run([lua, '-', str(rule)], input='''
o = {window = function(match, rule)
  local prefix = ({['^SussurroBar$'] = '', ['^SussurroMeeting$'] = 'meeting.'})[match.class]
  if prefix then
    for _, key in ipairs({'no_focus', 'no_initial_focus', 'no_follow_mouse',
                         'float', 'pin', 'render_unfocused'}) do
      print(prefix .. key .. '=' .. tostring(rule[key] or false))
    end
    print(prefix .. 'size=' .. rule.size[1] .. ',' .. rule.size[2])
    print(prefix .. 'rounding=' .. rule.rounding)
  end
end}
local handlers = {}
hl = {
  window_rule = function(rule)
    if not hyprbars_loaded and rule['hyprbars:no_bar'] ~= nil then
      error("hl.window_rule: unknown field 'hyprbars:no_bar'")
    end
    print('nobar.' .. rule.match.class .. '=' .. tostring(rule['hyprbars:no_bar']))
  end,
  on = function(event, fn) handlers[event] = fn end,
}
hyprbars_loaded = false
dofile(arg[1])
print('nobar_before_plugin=ok')
hyprbars_loaded = true
handlers['window.open']({class = 'Ghostty'})
handlers['window.open']({class = 'SussurroMeeting'})
handlers['window.open']({class = 'SussurroBar'})
''', text=True, capture_output=True, check=True)
        cls.output = result.stdout
        cls.rule = dict(line.split('=', 1) for line in result.stdout.splitlines())

    def test_pointer_hit_test_does_not_skip_bar(self):
        # Hyprland v0.56.2 ViewHitTester.cpp:57 exclui janelas pin com no_focus.
        self.assertEqual(self.rule['no_focus'], 'false')

    def test_opening_and_hovering_do_not_take_keyboard_focus(self):
        self.assertEqual(self.rule['no_initial_focus'], 'true')
        self.assertEqual(self.rule['no_follow_mouse'], 'true')

    def test_placement_and_rendering_policy_are_preserved(self):
        for key in ('float', 'pin', 'render_unfocused'):
            self.assertEqual(self.rule[key], 'true')
        self.assertEqual(self.rule['size'], '152,40')
        self.assertEqual(self.rule['rounding'], '20')

    def test_titlebars_of_floating_workspaces_skip_bar_and_meeting(self):
        # Sem o plugin o arquivo carrega inteiro; com ele, a regra entra uma vez so.
        self.assertEqual(self.rule['nobar_before_plugin'], 'ok')
        self.assertEqual(self.rule['nobar.^Sussurro(Bar|Meeting)$'], 'true')
        self.assertEqual(self.output.count('nobar.'), 1)

    def test_meeting_indicator_takes_clicks_without_stealing_focus(self):
        self.assertEqual(self.rule['meeting.no_focus'], 'false')
        self.assertEqual(self.rule['meeting.no_initial_focus'], 'true')
        self.assertEqual(self.rule['meeting.size'], f"{app.MeetingPill.W},{app.MeetingPill.H}")
        self.assertEqual(self.rule['meeting.rounding'], str(app.MeetingPill.H // 2))
        self.assertEqual(self.rule['size'], f"{app.RecorderBar.W},{app.RecorderBar.H}")


if __name__ == '__main__':
    unittest.main()
