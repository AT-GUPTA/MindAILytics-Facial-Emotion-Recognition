import os
from PIL import Image, ImageOps
import pandas as pd
from tkinter import *
from PIL import Image, ImageTk

def gather_image_data(root_dir='./Datasets'):
    data = []

    # Loop through each dataset directory
    for dataset_name in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_name)

        if os.path.isdir(dataset_path):
            # Loop through each emotion directory in the dataset
            for emotion in os.listdir(dataset_path):
                emotion_path = os.path.join(dataset_path, emotion)

                if os.path.isdir(emotion_path):
                    # Loop through each image in the emotion directory
                    for image_name in os.listdir(emotion_path):
                        image_path = os.path.join(emotion_path, image_name)

                        if os.path.isfile(image_path):
                            with Image.open(image_path) as img:
                                width, height = img.size
                            file_size = os.path.getsize(image_path)

                            # Append the data to the list
                            if emotion == 'focused':
                                emotion = 0
                            if emotion == 'happy':
                                emotion = 1
                            if emotion == 'neutral':
                                emotion = 2
                            if emotion == 'surprised':
                                emotion = 3

                            data.append({
                                'image_path': image_path,
                                'emotion': emotion,
                                'dataset_name': dataset_name,
                                'width': width,
                                'height': height,
                                'file_size': file_size,
                                'age': -1,
                                'gender':-1
                            })

    # Convert to a pandas DataFrame
    df = pd.DataFrame(data)
    return df

index = 0

class imgLabeler:
    def __init__(self, path, index, gender, age) -> None:
        self.img_path = path
        self.index = index
        self.gender = gender
        self.age = age

    def img_path(self) :
        return self.img_path
    def img_path(self, img_path):
        self.img_path = img_path
    def gender(self):
        return self.gender
    def gender(self, gender):
        self.gender = gender
    def age(self):
        return self.age
    def age(self, age):
        self.age = age
    def index(self):
        return self.index
    def index(self, index):
        self.index = index

def update_image(scrolling = False):

    global index
   
    gender_radio_val = 1
    age_radio_val = 1

    # if not scrolling:
    #     index += 1
    # else:
    #     if imgs[index].gender != -1:
    #         gender_radio_val = df.at[index, 'gender']
    #     if imgs[index].age != -1:
    #         age_radio_val = df.at[index, 'age']
    
    # update_label()
        
    # gender_var.set(gender_radio_val)
    # age_var.set(age_radio_val)

    if scrolling:
        if imgs[index].gender != -1:
            gender_radio_val = df.at[index, 'gender']
        if imgs[index].age != -1:
            age_radio_val = df.at[index, 'age']
    else:
        index = (index + 1) % len(imgs)
    
    update_label()

    gender_var.set(gender_radio_val)
    age_var.set(age_radio_val)

    next_image_path = imgs[index].img_path
    
    new_image = Image.open(next_image_path)
    new_image = new_image.resize((640, 640))
    new_photo = ImageTk.PhotoImage(new_image)
    image_label.configure(image=new_photo)
    image_label.image = new_photo

def on_closing():
    global df
    global df_all
    savingCSV(df_all, df, csv_file)
    print("Window is closing")
    root.destroy()

def savingCSV(df_all, df, path='./Datasets/datasets.csv'):
    if len(df) != len(df_all):
        for _, row in df.iterrows():
            original_index = row['original_index']
            df_all.loc[original_index, 'age'] = row['age']
            df_all.loc[original_index, 'gender'] = row['gender']
    df_all.to_csv(path, index=False)

def loadingDF( csv_file, path='./Datasets'):
    csv_file_path = csv_file
    df = None
    if os.path.exists(csv_file_path):
        print(f'Reading from CSV file')
        df = pd.read_csv(csv_file_path)
    else:
        print(f'Gathering images')
        df = gather_image_data(path)
    return df

def next_picture():
    global index
    index = (index + 1) % len(imgs)
    update_image(True)

def previous_picture():
    global index
    index = (index - 1) % len(imgs)
    update_image(True)

def on_enter(event):
    submit_action()

def shortcutsActions(event):
    if event.keysym == "Right":
        next_picture()
    if event.keysym == "Left":
        previous_picture()
    if event.keysym == 'f':
        gender_var.set(1)
    if event.keysym == 'm':
        gender_var.set(2)
    if event.keysym == 'x':
        gender_var.set(3)
    if event.keysym == 'y':
        age_var.set(1)    
    if event.keysym == 'a':
        age_var.set(2)
    if event.keysym == 'o':
        age_var.set(3)

def update_label():
    label.configure(text=f'[{index + 1}/{len(df.index)}]')

def submit_action():

    global index

    gender_radio_val = gender_var.get()
    age_radio_val = age_var.get()

    imgs[index].age = age_radio_val
    imgs[index].gender = gender_radio_val

    df.at[index, 'age'] = age_radio_val
    df.at[index, 'gender'] = gender_radio_val

    update_image(False)


csv_file = './Datasets/datasets.csv'
dataset = './Datasets/'

df_all = loadingDF(csv_file, dataset)
df = None

testing = False

if not testing:
    df = df_all[(df_all['age'] == -1) | (df_all['gender'].isin([-1, 3]))]
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'original_index'}, inplace=True)
else:
    df = df_all

print(f'{df_all.head()}')
print(f'{df.head()}')

imgs = []



for idx in range(df['image_path'].count()):
    if testing:
        img = imgLabeler(df.at[idx, 'image_path'], idx, df.at[idx, 'age'], df.at[idx, 'gender'])
        imgs.append(img)
    else:
        if df.at[idx, 'age'] == -1 or (df.at[idx, 'gender'] == -1 or df.at[idx, 'gender'] == 3):
            img = imgLabeler(df.at[idx, 'image_path'], idx, df.at[idx, 'age'], df.at[idx, 'gender'])
            imgs.append(img)
        # else:
        #     print(f'{df.iloc[idx].values}')

# print(f'{len(imgs)=}')

root = Tk()
root.title("Labeling")
root.geometry('1000x700')

gender_frame = LabelFrame(root, text="Gender", padx=20, pady=20)
gender_frame.grid(row=0, column=0, padx=10, pady=10, sticky="new")

age_frame = LabelFrame(root, text="Age", padx=20, pady=20)
age_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

previous_button = Button(root, text="<-", command=previous_picture)
previous_button.grid(row=2, column=1, pady=10, sticky='e')

label = Label(root, text=f'[{index + 1}/{len(imgs)}]')
label.grid(row=2, column=2, columnspan=2, sticky='n')

image_frame = Frame(root, width=640, height=640)
image_frame.grid(row=0, column=2, rowspan=2, columnspan=2, padx=10, pady=10, sticky="ew")

next_button = Button(root, text="->", command=next_picture)
next_button.grid(row=2, column=4, pady=10, sticky='w')

submit_button = Button(root, text="Submit", command=submit_action)
submit_button.grid(row=0, column=0, pady=20, sticky="sew")

gender_var = IntVar()


Radiobutton(gender_frame, text="Female", variable=gender_var, value=1).pack(anchor='w')
Radiobutton(gender_frame, text="Male", variable=gender_var, value=2).pack(anchor='w')
Radiobutton(gender_frame, text="Other", variable=gender_var, value=3).pack(anchor='w')

age_var = IntVar()


gender_radio_val = 1
age_radio_val = 1

print(f'{imgs[index].gender}, {imgs[index].age}')

if imgs[index].gender != -1:
        gender_radio_val = df.at[index, 'gender']
if imgs[index].age != -1:
    age_radio_val = df.at[index, 'age']

gender_var.set(gender_radio_val) 
age_var.set(age_radio_val)

Radiobutton(age_frame, text="Young Adult", variable=age_var, value=1).pack(anchor='w')
Radiobutton(age_frame, text="Middle Aged Adult", variable=age_var, value=2).pack(anchor='w')
Radiobutton(age_frame, text="Old-aged Adult", variable=age_var, value=3).pack(anchor='w')

image_path = imgs[index].img_path
image = Image.open(image_path)

photo = ImageTk.PhotoImage(image)

image_label = Label(image_frame, image=photo)
image_label.pack(fill="both", expand=True)




# root.columnconfigure(0, weight=0)
# root.columnconfigure(1, weight=0)
# root.columnconfigure(2, weight=0)
# root.columnconfigure(3, weight=0)
# root.columnconfigure(4, weight=0)
# root.rowconfigure(0, weight=0)
# root.rowconfigure(1, weight=0)
# root.rowconfigure(2, weight=0)
# root.rowconfigure(3, weight=0)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.bind('<Return>', on_enter)

root.bind('<Right>', shortcutsActions)
root.bind('<Left>', shortcutsActions)

root.bind('<f>', shortcutsActions)
root.bind('<m>', shortcutsActions)
root.bind('<x>', shortcutsActions)

root.bind('<y>', shortcutsActions)
root.bind('<a>', shortcutsActions)
root.bind('<o>', shortcutsActions)

root.mainloop()