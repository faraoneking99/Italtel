import prelearn_phase
import learn_phase
import try_predictions


def main():
    epochs = input("Specify the number of epochs: ")
    status, num_classi = prelearn_phase.main()
    
    if status == "pre-learn phase completed":
        print(status)

        if num_classi == 0:
            print("no classes were found in dataset folder, aborting learning...")

        else:
            
            status = learn_phase.main(epochs)
            print("---")
            print(status)
            choice = input("Would you like to execute a manual test with your picture? [y/n]")
            if choice == 'y':
                try_predictions.main()
            else:
                return status
    else:
        print("error during pre-learn phase, check dataset folder, each class must have a different folder.")



if __name__ == '__main__':
    main()
