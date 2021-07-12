from trainer import Trainer
from depencencies import installAll
# Installs dependencies if they haven't been installed yet
installAll() 

number_of_sessions = 20
infinite_train = True

if __name__ == "__main__":

    trainer = Trainer(num_iterations=10)

    counter = 1

    while(counter <= number_of_sessions or infinite_train):
        print(f"\n\n##################################\nTraining Session {counter}\n##################################\n\n")
        trainer.train_iteration()
        counter+=1

    trainer.cleanup()
    