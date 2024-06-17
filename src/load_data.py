import pandas as pd
import numpy as np
import mne
from sklearn.model_selection import train_test_split, KFold

class LoadData:
    def __init__(self, root, experiment_type: int, test_size: float = 0.2, random_state: int = 42, k_folds: int = 5):
        """
        ROOT: path to the root directory of the dataset

        Experiment type:
            1: Eyes Closed
            2: Eyes open
            3: Both
        """
        self.DATA_ROOT = root
        self.PARTICIPANTS_TSV = f'{self.DATA_ROOT}/participants.tsv'
        self.experiment_type = experiment_type

        # get class labels
        self.raw_labels = self.get_classes()
        self.labels = []

        # create data examples
        self.raw_data = []
        self.data = []
        self._create_eeg_objects()

        # holdout information
        self.test_size = test_size
        self.random_state = random_state

        # cross-validation information
        self.k_folds = k_folds
        self.split_participants = []

        # train and test data
        self.test = None
        self.train = None

        # split data
        self._cross_validation()
    

    def get_classes(self):
        participant_df = pd.read_csv(self.PARTICIPANTS_TSV, sep='\t')
        session_order = participant_df[['participant_id', 'SessionOrder']]

        # create ses-1 column based on the first two characters of the session order
        session_order['ses-1'] = session_order['SessionOrder'].str[:2]
        session_order['ses-2'] = session_order['SessionOrder'].str[4:]

        session_order = session_order.drop(columns=['SessionOrder'])

        session_order['ses-1'] = session_order['ses-1'].apply(lambda x: 1 if x == 'SD' else 0)
        session_order['ses-2'] = session_order['ses-2'].apply(lambda x: 1 if x == 'SD' else 0)

        return session_order
    
    def _create_eeg_objects(self): 
        def load_raw_eeg(example, root, participant_id, session, task, class_label):
            try:
                aux_eeg = mne.io.read_raw_eeglab(f"{root}/{participant_id}/{session}/eeg/{participant_id}_{session}_task-{task}_eeg.set").get_data()
                if class_label:
                    example[f"{task}_sd"] = (aux_eeg, class_label)
                else:
                    example[f"{task}_ns"] = (aux_eeg, class_label)
            except FileNotFoundError:
                print(f"For participant {participant_id}, the file {participant_id}_{session}_task-{task}_eeg.set was not found")
                aux_eeg = None
            except RuntimeError as e:
                print(f"Error loading participant {participant_id} with session {session} and task {task}: {e}")
                aux_eeg = None

            return example

        session_order = self.raw_labels

        for participant_id, ses_1, ses_2 in zip(session_order['participant_id'], session_order['ses-1'], session_order['ses-2']):
            example = dict()
            if self.experiment_type == 1:
                example = load_raw_eeg(example, self.DATA_ROOT, participant_id, 'ses-1', 'eyesclosed', ses_1)
                example = load_raw_eeg(example, self.DATA_ROOT, participant_id, 'ses-2', 'eyesclosed', ses_2)
            elif self.experiment_type == 2:
                example = load_raw_eeg(example, self.DATA_ROOT, participant_id, 'ses-1', 'eyesopen', ses_1)
                example = load_raw_eeg(example, self.DATA_ROOT, participant_id, 'ses-2', 'eyesopen', ses_2)
            else:
                example = load_raw_eeg(example, self.DATA_ROOT, participant_id, 'ses-1', 'eyesclosed', ses_1)
                example = load_raw_eeg(example, self.DATA_ROOT, participant_id, 'ses-2', 'eyesclosed', ses_2)
                example = load_raw_eeg(example, self.DATA_ROOT, participant_id, 'ses-1', 'eyesopen', ses_1)
                example = load_raw_eeg(example, self.DATA_ROOT, participant_id, 'ses-2', 'eyesopen', ses_2)

            if len(example.keys()):
                print(example.keys())
                example['participant_id'] = participant_id

            self.raw_data.append(example) if example.keys() else None
        
        self.raw_data = np.array(self.raw_data)  


    def _holdout(self, participant_id, participant_index):
        X_train, X_test, y_train, y_test = train_test_split(participant_id, participant_index, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test

    def _cross_validation(self):
        participant_id = np.array([x['participant_id'] for x in self.raw_data])
        participant_index = np.arange(len(participant_id))

        _, _, y_train, y_test_id = self._holdout(participant_id, participant_index)
        
        # save test data
        X_test = [self.raw_data[i][col][0] for i in y_test_id for col in self.raw_data[i].keys() if col != 'participant_id']
        y_test = np.array([self.raw_data[i][col][1] for i in y_test_id for col in self.raw_data[i].keys() if col != 'participant_id'])
        self.test = (X_test, y_test)

        # create splits
        skf = KFold(n_splits=self.k_folds)
        self.split_participants = list(skf.split(y_train))


    def get_split_data(self, split):
        split_ids = self.split_participants[split]
        train_ids, val_ids = split_ids

        X_train =[self.raw_data[i][col][0] for i in train_ids for col in self.raw_data[i].keys() if col != 'participant_id']
        y_train = np.array([self.raw_data[i][col][1] for i in train_ids for col in self.raw_data[i].keys() if col != 'participant_id'])
                
        X_val = [self.raw_data[i][col][0] for i in val_ids for col in self.raw_data[i].keys() if col != 'participant_id']
        y_val = np.array([self.raw_data[i][col][1] for i in val_ids for col in self.raw_data[i].keys() if col != 'participant_id'])

        return X_train, X_val, y_train, y_val

    def __len__(self):
        return len(self.raw_data)