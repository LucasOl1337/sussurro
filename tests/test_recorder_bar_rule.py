"""A barra precisa receber ponteiro, sem tomar foco ao abrir ou passar o mouse."""
import shutil
import subprocess
import unittest
from pathlib import Path


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
  if match.class == '^SussurroBar$' then
    for _, key in ipairs({'no_focus', 'no_initial_focus', 'no_follow_mouse',
                         'float', 'pin', 'render_unfocused'}) do
      print(key .. '=' .. tostring(rule[key] or false))
    end
    print('size=' .. rule.size[1] .. ',' .. rule.size[2])
    print('rounding=' .. rule.rounding)
  end
end}
dofile(arg[1])
''', text=True, capture_output=True, check=True)
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


if __name__ == '__main__':
    unittest.main()
