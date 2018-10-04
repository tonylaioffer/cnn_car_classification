import scipy.io
import shutil
import os


mat = scipy.io.loadmat('data/devkit/cars_train_annos.mat')
# print(mat['annotations'])
training_class = mat['annotations']['class']
training_fname = mat['annotations']['fname']
training_x1 = mat['annotations']['bbox_x1']
training_y1 = mat['annotations']['bbox_y1']
training_x2 = mat['annotations']['bbox_x2']
training_y2 = mat['annotations']['bbox_y2']

mat = scipy.io.loadmat('data/devkit/cars_test_annos_withlabels.mat')
print(mat['annotations'])
testing_class = mat['annotations']['class']
testing_fname = mat['annotations']['fname']
# print(testing_fname)
# print(testing_class)

training_source = 'cars_train/' # specify source training image path
training_output = 'train/' # specify target trainig image path (trainig images need to be orgnized to specific structure)
for idx, cls in enumerate(training_class[0]):
    cls = cls[0][0]
    fname = training_fname[0][idx][0]
    # print(cls)
    output_path = os.path.join(training_output, str(cls))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    shutil.copy(os.path.join(training_source, fname), os.path.join(output_path, fname))

testing_source = 'cars_test/' # specify source testing image path
testing_output = 'test/' # specify target testing image path (testing images need to be orgnized to specific structure)
for idx, cls in enumerate(testing_class[0]):
    cls = cls[0][0]
    fname = testing_fname[0][idx][0]
    output_path = os.path.join(testing_output, str(cls))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    shutil.copy(os.path.join(testing_source, fname), os.path.join(output_path, fname))