import unittest
import numpy as np
import pandas as pd

import cr_analysis as cra
import cr_interface as cri
from core import fine_model
import core.history as ch


class IoTest(unittest.TestCase):
    INITIAL_EPOCHS = 1

    @classmethod
    def setUpClass(cls):
        cls.fm = fine_model.FineMobileNetA25()
        cls.collection: cri.CrCollection = cri.CrCollection.load().sample(
            frac=0.05).tri_label().labeled()
        cls.images = cls.collection.load_images(
            target_size=cls.fm.get_output_shape(), stack=True)
        cls.generator = cls.fm.get_test_generator(cr_collection=cls.collection,
                                                  parent_dir='temp_images',
                                                  batch_size=16,
                                                  shuffle=False,
                                                  verbose=1)

        # Train model and predict
        cls.fm.compile_model()
        cls.fit_output = cls.fm.get_model().fit_generator(
            cls.generator,
            epochs=cls.INITIAL_EPOCHS,
            steps_per_epoch=len(cls.generator))
        cls.fm.save_weights(instance_key='test', exp_key='test')
        cls.predictions = cls.fm.get_model().predict(cls.images)

    def test_model_io(self):
        fm = self.__class__.fm
        predictions = self.__class__.predictions
        images = self.__class__.images

        # Reload weights and predict again
        fm.reload_model()
        fm.load_weights(instance_key='test', exp_key='test')
        new_predictions = fm.get_model().predict(images)

        # Assert that predictions are identical
        self.assertTrue(np.array_equal(predictions, new_predictions))

    def test_result_io(self):
        fm = self.__class__.fm
        predictions = self.__class__.predictions
        cr_codes = self.__class__.collection.get_cr_codes()

        result = cra.Result.from_predictions(predictions,
                                             cr_codes,
                                             params=dict(),
                                             short_name='test')
        result.save(model_key=fm.get_key(),
                    instance_key='test',
                    exp_key='test')
        loaded_result = cra.Result.load(model_key=fm.get_key(),
                                        instance_key='test',
                                        exp_key='test')
        self.assertTrue(result.df.equals(loaded_result.df))

    def test_histroy_io(self):
        fm = self.__class__.fm
        history = pd.DataFrame(self.__class__.fit_output.history)

        ch.save_history(history,
                        model_key=fm.get_key(),
                        instance_key='test',
                        exp_key='test')
        loaded_history = ch.load_history(model_key=fm.get_key(),
                                         instance_key='test',
                                         exp_key='test')
        self.assertEqual(len(history), self.__class__.INITIAL_EPOCHS)
        self.assertAlmostEqual(history.iloc[0, 0], loaded_history.iloc[0, 0])
        self.assertEqual(history.shape, loaded_history.shape)
        self.assertTrue(np.allclose(history, loaded_history))

    def test_history_append(self):
        epochs = 1
        fm = self.__class__.fm
        generator = self.__class__.generator

        history = ch.load_history(model_key=fm.get_key(),
                                  instance_key='test',
                                  exp_key='test')

        # Generate more history
        fm.compile_model()
        new_output = fm.get_model().fit_generator(
            generator, epochs=epochs, steps_per_epoch=len(generator))
        new_history = pd.DataFrame(new_output.history)

        # Append & load history
        ch.save_history(history,
                        model_key=fm.get_key(),
                        instance_key='test',
                        exp_key='test')
        ch.append_history(new_history,
                          model_key=fm.get_key(),
                          instance_key='test',
                          exp_key='test')
        appended = ch.load_history(model_key=fm.get_key(),
                                   instance_key='test',
                                   exp_key='test')

        # Compare manually appended history and auto-appended history
        manual_appended = pd.concat([history, new_history])

        total_epochs = self.__class__.INITIAL_EPOCHS + epochs
        self.assertEqual(len(appended), total_epochs)
        self.assertEqual(len(manual_appended), total_epochs)

        self.assertTrue(np.allclose(appended, manual_appended))

    def test_history_reset(self):
        history = pd.DataFrame(self.__class__.fit_output.history)
        fm = self.__class__.fm

        ch.save_history(history,
                        fm.get_key(),
                        instance_key='test',
                        exp_key='test')
        loaded_history = ch.load_history(fm.get_key(),
                                         instance_key='test',
                                         exp_key='test')

        ch.reset_history(fm.get_key(), instance_key='test', exp_key='test')
        reloaded_history = ch.load_history(fm.get_key(),
                                           instance_key='test',
                                           exp_key='test')

        self.assertTrue(np.allclose(history, loaded_history))
        self.assertEqual(reloaded_history, None)


if __name__ == '__main__':
    unittest.main()
