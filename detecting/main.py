import action_classifier
import rutube_downloader
import dataset_manager
import video_rescaller
import fittser

# region video

# https://rutube.ru/video/58fc540b4e1b1b59fe3e401865c04768/
# https://rutube.ru/video/b858a1dee8291420fa84076908fb3eaa/
# https://rutube.ru/video/686124d7b3523f705d1b06148889cf84/
video_url = "https://rutube.ru/video/686124d7b3523f705d1b06148889cf84/"

baza_name = "baza3.mp4"
input_path = 'detecting/rutube_d/' + baza_name
rescaled_path = 'detecting/rescaled/' + baza_name
output_path = 'detecting/rescaled/' + baza_name

#rutube_downloader.download_rutube_video(video_url, input_path)

new_width = 1024
new_height = 1024

#video_rescaller.change_video_resolution(input_path, rescaled_path, new_width, new_height)

# endregion

# region dataset

dataset_path = "detecting/datasets/UFC-CRIME"
dataset_loader = dataset_manager.DatasetManager(dataset_path)

labels = dataset_loader.get_labels()

# endregion

model_weights = "yolo11n.pt"
# https://huggingface.co/microsoft/xclip-base-patch32
# microsoft/xclip-base-patch16-kinetics-600-16-frames
# microsoft/xclip-base-patch32
video_classifier_model =  "microsoft/xclip-base-patch16-kinetics-600"

fit_path = fittser.fit(num_epochs=1, video_classifier_model=video_classifier_model)
#print(fit_path)

#action_classifier.run(device = "cuda", video_classifier_model=video_classifier_model, source=input_path, output_path=output_path, labels=labels, weights=model_weights)
