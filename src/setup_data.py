import pathlib
import shutil
import os

WALDO_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../Hey-Waldo/'


def count_files(img_dir):
    return len([1 for x in list(os.scandir(img_dir + '/waldo/')) if x.is_file()]),\
            len([1 for x in list(os.scandir(img_dir + '/notwaldo/')) if x.is_file()])


def organize_data(img_dir):
    nb_w, nb_nw = count_files(img_dir)
    cnt = 0

    nb_test_w = nb_w / 10
    nb_test_nw = nb_nw / 10

    for fname in os.listdir(img_dir + '/waldo/'):
        shutil.move(img_dir + '/waldo/' + fname,
                    img_dir + '/test/waldo/' + fname)
        cnt += 1
        if cnt >= nb_test_w:
            break
    for fname in os.listdir(img_dir + '/waldo/'):
        shutil.move(img_dir + '/waldo/' + fname,
                    img_dir + '/train/waldo/' + fname)

    cnt = 0
    for fname in os.listdir(img_dir + '/notwaldo/'):
        shutil.move(img_dir + '/notwaldo/' + fname,
                    img_dir + '/test/notwaldo/' + fname)
        cnt += 1
        if cnt >= nb_test_nw:
            break
    for fname in os.listdir(img_dir + '/notwaldo/'):
        shutil.move(img_dir + '/notwaldo/' + fname,
                    img_dir + '/train/notwaldo/' + fname)

    os.rmdir(img_dir + '/waldo')
    os.rmdir(img_dir + '/notwaldo')


def create_directories(img_dir):
    pathlib.Path(img_dir + '/train').mkdir(exist_ok=False)
    pathlib.Path(img_dir + '/test').mkdir(exist_ok=False)

    pathlib.Path(img_dir + '/train/waldo').mkdir(exist_ok=False)
    pathlib.Path(img_dir + '/train/notwaldo').mkdir(exist_ok=False)
    pathlib.Path(img_dir + '/test/waldo').mkdir(exist_ok=False)
    pathlib.Path(img_dir + '/test/notwaldo').mkdir(exist_ok=False)


if __name__ == '__main__':
    for file in os.listdir(WALDO_DIR):
        if file != 'original-images' and os.path.isdir(os.path.join(WALDO_DIR, file)):
            create_directories(os.path.join(WALDO_DIR, file))
            organize_data(os.path.join(WALDO_DIR, file))
