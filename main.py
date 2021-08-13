from helper import convert_list_to_dash_separated
import os
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from visualization import Visualization
from trainer import Trainer, delete_progess
from depencencies import installAll
# Installs dependencies if they haven't been installed yet
installAll() 

number_of_sessions = 3
infinite_train = True
evaluate_learning_rate=False
# learning_rates = [0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008]
# learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
learning_rates = [0.00000997]
# learning_rates = [0.0001, 0.00001, 0.000001, 0.0000001]
# learning_rates = [0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
# learning_rates = [0.000000003, 0.000000004, 0.000000005, 0.000000006, 0.000000007, 0.000000008, 0.000000009, 0.00000001]
# learning_rates = [0.000000001, 0.000000002, 0.000000003, 0.000000004]
# learning_rates = [0.0000000027 ,0.0000000028, 0.0000000029, 0.000000003, 0.0000000031, 0.0000000032, 0.0000000033]
# learning_rates = [0.00000000291 ,0.00000000292, 0.00000000293, 0.00000000294, 0.00000000295, 0.00000000296, 0.00000000297, 0.00000000298]
chosen_fc_layer_params = (100, 100, 100, 100)
fc_layer_params_list = [(100, 100, 100, 100)]
# fc_layer_params_list = [(4610, 100), (512, 256), (256, 128, 64), (922, 256, 128)]
display_chart=False
chart_x_axis_logarithmic_Loss_vs_param = False


if __name__ == "__main__":

    if(not evaluate_learning_rate):

        trainer = Trainer(num_iterations=300, learning_rate=0.00000997, collect_steps_per_iteration=200, window_scale=0.1, display_visualization=False, show_chart=False, fc_layer_params = chosen_fc_layer_params)

        counter = 1

        while(counter <= number_of_sessions or infinite_train):
            print(f"\n\n##################################\nTraining Session {counter}\n##################################\n\n")
            trainer.train_iteration()
            counter+=1

        trainer.cleanup()

    else:

        # generate charts folder
        chart_path = os.path.join(os.getcwd(), "charts")
        if not os.path.exists(chart_path):
            os.makedirs(chart_path)

        for fc_layer_params_current in fc_layer_params_list:

            print("\n\n\nLayer Params: " + str(fc_layer_params_current) + "\n\n\n")

            avg_losses = []

            for learning_rate in learning_rates:

                print(f"\n\n###########################################################\nTraining at learning rate of {learning_rate} in {learning_rates}\n###########################################################\n\n")


                trainer = Trainer(num_iterations=300, learning_rate=learning_rate, collect_steps_per_iteration=200, window_scale=0.1, display_visualization=False, show_chart=False, fc_layer_params=fc_layer_params_current)

                delete_progess(trainer)

                trainer.train_iteration()

                avg_losses.append(np.mean(trainer.losses))

                trainer.cleanup()

            print(f"learning rates: {learning_rates}\navg_losses: {avg_losses}")

            plt.plot(learning_rates, avg_losses)
            plt.ylabel('Average Loss')
            plt.xlabel('learning rates')
            if chart_x_axis_logarithmic_Loss_vs_param:
                plt.xscale("log")
            plt.ylim(top=np.max(avg_losses))

            file_name = "learning-rates-"

            file_name = file_name + convert_list_to_dash_separated(learning_rates)

            file_name = "-" + file_name + "["

            for param in fc_layer_params_current:
                file_name = file_name + convert_list_to_dash_separated(param)

            file_name = file_name + "]-"

            file_name = file_name + "chart.png"

            plt.savefig(os.path.join(chart_path, file_name))
            if(display_chart):
                plt.show()


    