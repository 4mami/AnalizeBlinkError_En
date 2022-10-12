import unittest
import PreProcessCirrusDump

class MyTest(unittest.TestCase):
    def test_RoughenCategory_1(self):
        old_categories = list(map(lambda c: c.lower(), [
            'Scooby-Doo', 
            'Fictional characters introduced in 1969', 
            'Media franchises', 
            'Television programs adapted into films', 
            'Television programs adapted into comics'
        ]))
        self.assertEqual(sorted(PreProcessCirrusDump.RoughenCategory(old_categories)), [
            '1969', 
            'adapted', 
            'characters', 
            'comics', 
            'fictional', 
            'films', 
            'franchises', 
            'into', 
            'introduced', 
            'media', 
            'programs', 
            'scooby-doo', 
            'television'
        ])

    def test_RoughenCategory_2(self):
        old_categories = list(map(lambda c: c.lower(), [
            'Nose',
            'Human head and neck',
            'Respiratory system',
            'Olfactory system',
            'Facial features'
        ]))
        self.assertEqual(sorted(PreProcessCirrusDump.RoughenCategory(old_categories)), [
            'and',
            'facial',
            'features',
            'head',
            'human',
            'neck',
            'nose',
            'olfactory',
            'respiratory',
            'system'
        ])

if __name__ == '__main__':
    unittest.main()