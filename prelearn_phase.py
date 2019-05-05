import numpy as np
import shutil
import pickle
import os
import data_augmentation as augment


def main():
    aug = input("Would you like to perform data augmentation on the dataset? [y/n]:")
    if aug != 'y':
        print("Command not recognized, skipping data-augmentation...")
    if aug == 'y':
        num_files_desired = int(input("How many images for each class would you like to generate? "))

    print("pre-learn phase inizialized, this may take a while")

    dataset = 'dataset'

    model = 'model'

    test = 'test'
    train = 'train'
    valid = 'valid'

    classi = os.listdir(dataset)
    numClassi = len(os.listdir(dataset))
    print(classi)

    print(numClassi)
    immagini = []

    primo = []
    secondo = []
    terzo = []

    errori = []
    dir_name = "model"
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)

    os.mkdir(model)

    os.mkdir(os.path.join(model, test))
    os.mkdir(os.path.join(model, train))
    os.mkdir(os.path.join(model, valid))
    os.mkdir(os.path.join(model, 'config'))
    with open('model/config/config.cfg', 'wb') as fp:
        pickle.dump(classi, fp)

    size = []
    for classe in os.listdir(dataset):

        os.mkdir(os.path.join(model, test, classe))
        os.mkdir(os.path.join(model, train, classe))
        os.mkdir(os.path.join(model, valid, classe))

        i = 0

        print(classe)
        for filename in os.listdir(os.path.join(dataset, classe)):
            src = os.path.join(dataset, classe, filename)
            dst = os.path.join(dataset, classe, classe + str(i) + ".jpg")
            try:
                os.rename(src, dst)
            except(Exception):
                errori.append(filename)
            i += 1

        p = int(i / 100 * 80)  # train
        q = int(i / 100 * 10)  # test
        r = int(i / 100 * 10)  # validation

        print('train per ' + classe + " " + str(p))
        print('test per ' + classe + " " + str(q))
        print('valid per ' + classe + " " + str(r))

        immagini = os.listdir(os.path.join(dataset, classe))

        scelteTrain = np.random.choice(immagini, p, replace=False)
        size.append(len(scelteTrain))
        if aug == 'y':
            dst = os.path.join(model, train, classe)
            augment.augment(os.path.join(dataset, classe), num_files_desired, dst)
            for file in scelteTrain:
                src = os.path.join(dataset, classe, file)
                #shutil.copy2(src, dst)
                immagini.remove(file)
        else:
            for file in scelteTrain:
                src = os.path.join(dataset, classe, file)
                dst = os.path.join(model, train, classe)
                shutil.copy2(src, dst)
                immagini.remove(file)

        scelteTest = np.random.choice(immagini, q, replace=False)
        size.append(len(scelteTest))
        for file in scelteTest:
            src = os.path.join(dataset, classe, file)
            dst = os.path.join(model, test, classe)
            shutil.copy2(src, dst)
            immagini.remove(file)

        scelteValid = np.random.choice(immagini, r, replace=False)
        size.append(len(scelteValid))
        for file in scelteTest:
            src = os.path.join(dataset, classe, file)
            dst = os.path.join(model, valid, classe)
            shutil.copy2(src, dst)
            try:
                immagini.remove(file)
            except:
                errori.append(file)


        name = 'model/config/' + str(classe) + '.cfg'
        with open(name, 'wb') as fp:
            pickle.dump(size, fp)
    return "pre-learn phase completed", numClassi


if __name__ == '__main__':
    main()
