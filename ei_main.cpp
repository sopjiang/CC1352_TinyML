/* Edge Impulse ingestion SDK
 * Copyright (c) 2021 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* Include ----------------------------------------------------------------- */
#include "ei_run_classifier.h"
#include "ei_classifier_porting.h"
#include "numpy.hpp"

static const float features[] = {
    // copy raw features here (for example from the 'Live classification' page)
    // see https://docs.edgeimpulse.com/docs/running-your-impulse-ti-launchxl
    1.2073, 2.1159, 16.0669, 1.3216, 1.8944, 15.8101, 1.4772, 2.1841, 15.7880, 1.1205, 2.3212, 15.8783, 0.7667, 2.4756, 16.0687, 0.5243, 2.5863, 16.0298, 0.7075, 2.8880, 16.0112, 0.9936, 3.0807, 16.0196, 0.9559, 3.3980, 16.3129, 0.8230, 3.4590, 16.4859, 0.4531, 3.5195, 16.4218, 0.0892, 3.3866, 15.8299, 0.0048, 3.2250, 15.0721, 0.1179, 3.1478, 14.0552, 0.0874, 3.1603, 13.0807, 0.3549, 2.9634, 12.2583, 0.1957, 2.8228, 11.6873, 0.1790, 2.7515, 10.9074, 0.1137, 2.2158, 10.3334, 0.2173, 1.8166, 9.6929, -0.0048, 1.2558, 9.3901, -0.2376, 1.2558, 8.6006, -0.4214, 0.9152, 8.1397, -0.4962, 0.6919, 7.9464, -0.3885, 0.4250, 7.8254, -0.4304, 0.2717, 7.3095, -0.5686, -0.2173, 6.4542, -0.4597, -0.6357, 5.7239, -0.5644, -1.0672, 5.2547, -0.4220, -1.1612, 5.0344, -0.3867, -1.3186, 5.2044, -0.4100, -1.4832, 5.4558, -0.3244, -1.7675, 5.5432, -0.5956, -1.8902, 5.9245, -0.5321, -2.0584, 5.8012, -0.6692, -2.3182, 5.5276, -0.5339, -2.4169, 4.8920, -0.3675, -2.5145, 4.1797, -0.6009, -2.4565, 3.5686, -0.9517, -2.5001, 2.9012, -1.7322, -2.7282, 2.3972, -2.1452, -2.8695, 2.1614, -1.9938, -2.7336, 2.4948, -1.7202, -2.5666, 2.7132, -1.2354, -2.5343, 3.0795, -0.8829, -2.3655, 3.2016, -0.6979, -2.2206, 3.4477, -0.7242, -1.9573, 3.5560, -0.9930, -1.3844, 3.5291, -1.3749, -1.2288, 4.0205, -1.5700, -0.6788, 4.6070, -1.3946, -0.0329, 5.3241, -1.3114, 0.1766, 6.0328, -0.9254, 0.3891, 6.9300, -0.7099, 0.7398, 8.2516, -0.4615, 1.1157, 9.4248, -0.4770, 1.2917, 10.8433, -0.2622, 1.7286, 11.4557, -0.0078, 1.7759, 11.6316, -0.2705, 1.8364, 12.0117, -0.3867, 1.8058, 12.5259, -0.6865, 1.6143, 13.0813, -0.7506, 1.6879, 13.2890, -0.3974, 1.8118, 12.9431, -0.4800, 1.9549, 12.5318, -0.9158, 1.9309, 12.6288, -1.5778, 2.1919, 13.8205, -1.9758, 2.4122, 14.6106, -1.9088, 2.1129, 15.1655, -1.3012, 2.1290, 14.7213, -1.3150, 2.1524, 14.3455, -1.7615, 2.4068, 14.9296, -2.3828, 2.5546, 15.5845, -2.2835, 2.5013, 16.1686, -2.0506, 2.1991, 16.2087, -1.5658, 2.3703, 16.2231, -1.4204, 2.1865, 16.1680, -1.3048, 2.3176, 16.0394, -1.3868, 2.5582, 16.3548, -1.2947, 2.7054, 16.3416, -1.1121, 2.6498, 15.8730, -1.0983, 2.6043, 15.1230, -0.8422, 2.3260, 14.1258, -0.7320, 2.2895, 13.2657, -0.7787, 2.2326, 12.4427, -0.6285, 2.0991, 11.9285, -0.5321, 2.0273, 11.4072, -0.5453, 1.8711, 10.9008, -0.3657, 1.7370, 10.6823, -0.4202, 1.7603, 10.4238, -0.5465, 1.5550, 10.0287, -0.3394, 1.4425, 9.7193, -0.2646, 1.3336, 9.2871, -0.3723, 0.9996, 8.9795, -0.0389, 0.7314, 8.5874, 0.2029, 0.4956, 8.2911, 0.5351, 0.4112, 8.2576, 1.0469, 0.3322, 8.1457, 1.1696, 0.1784, 7.8260, 1.3366, 0.0419, 7.4855, 1.1067, -0.1347, 6.9085, 0.7392, -0.4525, 6.4039, 0.4094, -0.6237, 5.9047, 0.4609, -0.6590, 5.7898, 0.6500, -0.6859, 5.6132, 0.4806, -0.6039, 5.5462, 1.2605, -0.7284, 5.1649, 2.1261, -0.6632, 4.8842, 2.4349, -0.3573, 4.5699, 0.1448, -0.3041, 5.0170, -2.3774, -0.2747, 5.7832, -2.8910, -0.4304, 6.2782, -1.5574, -0.3448, 6.3458, -0.2628, -0.2161, 6.2321, -0.2053, -0.1299, 6.3650, -0.5357, 0.2167, 6.9911, -2.0153, 0.6476, 7.9296, -4.3844, 0.4106, 8.8442, -4.3485, 0.4705, 9.2524, -2.7719, 0.8260, 8.9735, -2.1715, 0.9140, 8.5246, -2.7288, 1.2175, 9.0950, -3.4650, 1.5347, 10.1203, -4.2653, 1.6245, 11.2569, -4.2994, 1.7232, 12.1428
};


int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

extern "C" int ei_main() {
    ei_printf("Edge Impulse standalone inferencing (TI LaunchXL)\n");

    if (sizeof(features) / sizeof(float) != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        ei_printf("The size of your 'features' array is not correct. Expected %d items, but had %u\n",
            EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(features) / sizeof(float));
        return 1;
    }

    ei_impulse_result_t result = { 0 };

    while (1) {
        // the features are stored into flash, and we don't want to load everything into RAM
        signal_t features_signal;
        features_signal.total_length = sizeof(features) / sizeof(features[0]);
        features_signal.get_data = &raw_feature_get_data;

        // invoke the impulse
        EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, true);


        ei_printf("run_classifier returned: %d\n", res);

        if (res != 0) return 1;

        ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);

        // print the predictions
        ei_printf("[");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("%.5f", result.classification[ix].value);
#if EI_CLASSIFIER_HAS_ANOMALY == 1
            ei_printf(", ");
#else
            if (ix != EI_CLASSIFIER_LABEL_COUNT - 1) {
                ei_printf(", ");
            }
#endif
        }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
        ei_printf("%.3f", result.anomaly);
#endif
        ei_printf("]\n");

        ei_sleep(2000);
    }
}
