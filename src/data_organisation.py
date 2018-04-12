import pathlib
import shutil
import sys
import os

WALDO_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../Hey-Waldo/'


def undo_organize_data(img_dir):
    pathlib.Path(WALDO_DIR + img_dir + '/waldo').mkdir(exist_ok=True)
    pathlib.Path(WALDO_DIR + img_dir + '/notwaldo').mkdir(exist_ok=True)

    for fname in os.listdir(WALDO_DIR + img_dir + '/train/waldo/'):
        shutil.move(WALDO_DIR + img_dir + '/train/waldo/' + fname,
                    WALDO_DIR + img_dir + '/waldo/' + fname)
    for fname in os.listdir(WALDO_DIR + img_dir + '/test/waldo/'):
        shutil.move(WALDO_DIR + img_dir + '/test/waldo/' + fname,
                    WALDO_DIR + img_dir + '/waldo/' + fname)
    for fname in os.listdir(WALDO_DIR + img_dir + '/train/notwaldo/'):
        shutil.move(WALDO_DIR + img_dir + '/train/notwaldo/' + fname,
                    WALDO_DIR + img_dir + '/notwaldo/' + fname)
    for fname in os.listdir(WALDO_DIR + img_dir + '/test/notwaldo/'):
        shutil.move(WALDO_DIR + img_dir + '/test/notwaldo/' + fname,
                    WALDO_DIR + img_dir + '/notwaldo/' + fname)

    os.rmdir(WALDO_DIR + img_dir + '/train/waldo')
    os.rmdir(WALDO_DIR + img_dir + '/train/notwaldo')
    os.rmdir(WALDO_DIR + img_dir + '/train')
    os.rmdir(WALDO_DIR + img_dir + '/test/waldo')
    os.rmdir(WALDO_DIR + img_dir + '/test/notwaldo')
    os.rmdir(WALDO_DIR + img_dir + '/test')


def count_files(img_dir):
    return len([1 for x in list(os.scandir(WALDO_DIR + img_dir + '/waldo/')) if x.is_file()]),\
            len([1 for x in list(os.scandir(WALDO_DIR + img_dir + '/notwaldo/')) if x.is_file()])


def organize_data(img_dir):
    nb_w, nb_nw = count_files(img_dir)
    cnt = 0

    nb_test_w = nb_w / 10
    nb_test_nw = nb_nw / 10

    for fname in os.listdir(WALDO_DIR + img_dir + '/waldo/'):
        shutil.move(WALDO_DIR + img_dir + '/waldo/' + fname,
                    WALDO_DIR + img_dir + '/test/waldo/' + fname)
        cnt += 1
        if cnt >= nb_test_w:
            break
    for fname in os.listdir(WALDO_DIR + img_dir + '/waldo/'):
        shutil.move(WALDO_DIR + img_dir + '/waldo/' + fname,
                    WALDO_DIR + img_dir + '/train/waldo/' + fname)

    cnt = 0
    for fname in os.listdir(WALDO_DIR + img_dir + '/notwaldo/'):
        shutil.move(WALDO_DIR + img_dir + '/notwaldo/' + fname,
                    WALDO_DIR + img_dir + '/test/notwaldo/' + fname)
        cnt += 1
        if cnt >= nb_test_nw:
            break
    for fname in os.listdir(WALDO_DIR + img_dir + '/notwaldo/'):
        shutil.move(WALDO_DIR + img_dir + '/notwaldo/' + fname,
                    WALDO_DIR + img_dir + '/train/notwaldo/' + fname)

    os.rmdir(WALDO_DIR + img_dir + '/waldo')
    os.rmdir(WALDO_DIR + img_dir + '/notwaldo')


def create_directories(img_dir):
    pathlib.Path(WALDO_DIR + img_dir + '/train').mkdir(exist_ok=True)
    pathlib.Path(WALDO_DIR + img_dir + '/test').mkdir(exist_ok=True)

    pathlib.Path(WALDO_DIR + img_dir + '/train/waldo').mkdir(exist_ok=True)
    pathlib.Path(WALDO_DIR + img_dir + '/train/notwaldo').mkdir(exist_ok=True)
    pathlib.Path(WALDO_DIR + img_dir + '/test/waldo').mkdir(exist_ok=True)
    pathlib.Path(WALDO_DIR + img_dir + '/test/notwaldo').mkdir(exist_ok=True)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        image_dir = sys.argv[2]
        if sys.argv[1] == 'build':
            create_directories(image_dir)
            organize_data(image_dir)
        elif sys.argv[1] == 'clear':
            undo_organize_data(image_dir)
    else:
        print('Usage: data_organisation.py ["build" or "clear"] [image directory (ex: "64-bw")]')
