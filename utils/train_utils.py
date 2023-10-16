import logging
import os
import time
import warnings
import torch
from torch import nn, optim
import models
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE


class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir
        self.criterion = None
        self.start_epoch = None
        self.optimizer = None
        self.lr_scheduler = None
        self.model = None
        self.dataloaders = None
        self.datasets = None
        self.device_count = None
        self.device = None

    def setup(self):
        self._setup_device()
        self._load_datasets()
        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_lr_scheduler()
        self._load_checkpoint()
        self._initialize_criterion()

    def _setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('Using {} GPUs'.format(self.device_count))
            assert self.args.batch_size % self.device_count == 0, "Batch size should be divided by device count"
        else:
            warnings.warn("GPU is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('Using {} CPU'.format(self.device_count))

    def _load_datasets(self):
        dataset = self._load_dataset()
        train_dataset, val_dataset, test_dataset = dataset(self.args.data_dir, self.args.normlizetype).data_prepare()

        self.datasets = {
            'Train': train_dataset,
            'Validation': val_dataset,
            'Test': test_dataset
        }

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=self.args.batch_size,
                shuffle=(True if x == 'Train' else False),
                num_workers=self.args.num_workers,
                pin_memory=(True if self.device == 'cuda' else False)
            ) for x in ['Train', 'Validation', 'Test']
        }

        num_of_train_sam = self.train_samples = len(self.dataloaders['Train'].dataset)
        num_of_valid_sam = self.validation_samples = len(self.dataloaders['Validation'].dataset)
        num_of_test_sam = self.testing_samples = len(self.dataloaders['Test'].dataset)

        print("Training Samples Count", num_of_train_sam)
        print("Validation Samples Count", num_of_valid_sam)
        print("Testing Samples Count", num_of_test_sam)

    def _load_dataset(self):
        processing_type = self.args.processing_type
        dataset_name = self.args.data_name

        if processing_type == 'O_A':
            from CNN_Datasets.O_A import datasets
        elif processing_type == 'R_A':
            from CNN_Datasets.R_A import datasets
        elif processing_type == 'R_NA':
            from CNN_Datasets.R_NA import datasets
        else:
            raise Exception("Processing Type Not Implemented")

        return getattr(datasets, dataset_name)

    def _initialize_model(self):
        dataset = self._load_dataset()
        self.model = getattr(models, self.args.model_name)(
            in_channel=dataset.inputchannel,
            out_channel=dataset.num_classes
        )
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

    def _initialize_optimizer(self):
        if self.args.opt == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr,
                                       momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr,
                                        weight_decay=self.args.weight_decay)
        elif self.args.opt == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.opt == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.args.lr, alpha=0.99, eps=1e-8,
                                           momentum=self.args.momentum,
                                           weight_decay=self.args.weight_decay)
        else:
            raise Exception("Optimizer not implemented")

    def _initialize_lr_scheduler(self):
        if self.args.lr_scheduler == 'step':
            steps = [int(step) for step in self.args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=self.args.gamma)
        elif self.args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.args.gamma)
        elif self.args.lr_scheduler == 'stepLR':
            steps = int(self.args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, self.args.gamma)
        elif self.args.lr_scheduler == 'reduce':
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                     factor=self.args.factor,
                                                                     patience=self.args.patience, verbose=True)
        elif self.args.lr_scheduler == 'cosine':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.T_max)
        elif self.args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("LR Schedule Not Implemented")

    def _load_checkpoint(self):
        self.start_epoch = 0

    def _initialize_criterion(self):
        self.model.to(self.device, dtype=torch.float32)
        # self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        args = self.args
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0

        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []

        total_time = 0  # Initialize the total time taken for training to 0

        # Calculate the DD:HH:MM:SS breakdown from the total_time in seconds
        days, remainder = divmod(total_time, 86400)  # 86400 seconds in a day
        hours, remainder = divmod(remainder, 3600)  # 3600 seconds in an hour
        minutes, seconds = divmod(remainder, 60)  # 60 seconds in a minute

        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 10 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 10)

            # Update the Learning Rate
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # For ReduceLROnPlateau, get lr from optimizer directly
                    lr = self.optimizer.param_groups[0]['lr']
                else:
                    # For other schedulers, use get_last_lr()
                    lr = self.lr_scheduler.get_last_lr()[0]
                logging.info('Current Learning Rate: {}'.format(lr))

            # Each Epoch Has a Training and Val Phase
            for phase in ['Train', 'Validation']:
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0

                if phase == 'Train':
                    self.model.train()
                else:
                    self.model.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device, dtype=torch.float32)
                    # inputs = inputs.to(self.device)
                    # print("Type of Input", inputs.type())
                    labels = labels.to(self.device)

                    with torch.set_grad_enabled(phase == 'Train'):
                        _, out = self.model(inputs)
                        loss = self.criterion(out, labels)
                        pred = out.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        if phase == 'Train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += inputs.size(0)

                            if batch_count == len(self.dataloaders[phase].dataset):
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count

                                train_loss_history.append(batch_loss)
                                train_acc_history.append(batch_acc)
                                batch_loss = 0.0
                                batch_acc = 0
                                batch_count = 0

                epoch_loss /= len(self.dataloaders[phase].dataset)
                epoch_acc /= len(self.dataloaders[phase].dataset)

                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Accuracy: {:.4f}, Cost {:.4f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start))

                total_time += time.time() - epoch_start  # Add the time taken for the current epoch to the total time

                if phase == 'Validation':
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    if epoch_acc > best_acc or epoch > args.max_epoch - 2:
                        best_acc = epoch_acc
                        logging.info("Save Best Model Epoch {}, Accuracy {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc)

                    # Adjust learning rate if you are using the ReduceLROnPlateau scheduler
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(epoch_loss)

            # For all other schedulers that don't require validation loss
            if self.lr_scheduler is not None and not isinstance(self.lr_scheduler,
                                                                torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step()

        # After the training loop, print the total time taken for training
        logging.info(
            'Total training time: {:.4f} Seconds ({} Days, {} Hours, {} Minutes, {:.2f} Seconds)'.format(total_time,
                                                                                                         int(days),
                                                                                                         int(hours),
                                                                                                         int(minutes),
                                                                                                         total_time % 60))

        # Testing Phase after training is complete
        self.model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        test_predictions = []  # Store model predictions
        test_labels = []  # Store ground truth labels

        logging.info('----------Starting Testing Phase----------')

        with torch.no_grad():
            # for batch_idx, (inputs, labels) in enumerate(self.dataloaders['Test']):
            for batch_idx, (inputs, labels) in enumerate(self.dataloaders['Test']):
                inputs = inputs.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device)

                _, out = self.model(inputs)
                loss = self.criterion(out, labels)
                pred = out.argmax(dim=1)

                test_loss += loss.item()  # Accumulate loss

                # Count correct predictions
                test_correct += (pred == labels).sum().item()
                test_total += labels.size(0)

                # Collect predictions and labels
                test_predictions.extend(pred.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        # Calculate final metrics
        test_loss /= len(self.dataloaders['Test'])
        test_accuracy = test_correct / test_total

        logging.info('Test-Loss: {:.4f} Test-Accuracy: {:.4f}'.format(test_loss, test_accuracy))

        # Automatically visualize the confusion matrix after training
        self.plot_loss_and_accuracy(train_loss_history, val_loss_history, train_acc_history, val_acc_history)
        self.visualize_normalized_confusion_matrix()
        self.visualize_tsne()

    def plot_loss_and_accuracy(self, train_loss_history, val_loss_history, train_acc_history, val_acc_history):

        # Define aesthetic properties
        font_name = "Palatino Linotype"
        title_font_size = 24
        font_size = 16
        line_width = 2.5

        def base_plot_settings(ax, xlabel, ylabel, title):
            """Base plot settings."""
            # Labels, title, and legend
            ax.set_xlabel(xlabel, fontname=font_name, fontsize=font_size, fontweight='bold', labelpad=10)
            ax.set_ylabel(ylabel, fontname=font_name, fontsize=font_size, fontweight='bold', labelpad=10)

            # Tick aesthetics
            plt.xticks(fontname=font_name, fontsize=font_size)
            plt.yticks(fontname=font_name, fontsize=font_size)

            # Grid settings
            plt.grid()

        # Plot Training Loss and Validation Loss
        plt.figure(figsize=(10, 8))
        plt.plot(np.arange(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss',
                 color='royalblue', linewidth=line_width)
        plt.plot(np.arange(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss', color='tomato',
                 linewidth=line_width)
        ax = plt.gca()
        base_plot_settings(ax, 'Epoch', 'Loss', 'Training Loss and Validation Loss')
        plt.legend(prop={'family': font_name, 'size': font_size}, loc='upper right', frameon=True, edgecolor='gray')
        plt.tight_layout()
        plt.show()

        # Plot Training Accuracy and Validation Accuracy
        plt.figure(figsize=(10, 8))
        plt.plot(np.arange(1, len(train_acc_history) + 1), train_acc_history, label='Training Accuracy',
                 color='royalblue',
                 linewidth=line_width)
        plt.plot(np.arange(1, len(val_acc_history) + 1), val_acc_history, label='Validation Accuracy', color='tomato',
                 linewidth=line_width)
        ax = plt.gca()
        base_plot_settings(ax, 'Epochs', 'Accuracy', 'Training Accuracy and Validation Accuracy')
        plt.legend(prop={'family': font_name, 'size': font_size}, loc='lower right', frameon=True, edgecolor='gray')
        plt.tight_layout()
        plt.show()

    def visualize_normalized_confusion_matrix(self):
        # Initialize lists to store ground truth and predictions
        ground_truth = []
        predictions = []

        for batch_idx, (inputs, labels) in enumerate(self.dataloaders['Test']):
            inputs = inputs.to(self.device, dtype=torch.float32)
            # inputs = inputs.to(self.device)

            # Forward pass to get predictions
            with torch.no_grad():
                features_map, out = self.model(inputs)
                pred = out.argmax(dim=1)

                # Append ground truth and predictions
                ground_truth.extend(labels.cpu().numpy())
                predictions.extend(pred.cpu().numpy())

        # Ensure that ground_truth and predictions have the same length
        assert len(ground_truth) == len(predictions), "Inconsistent Numbers of Samples"

        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(ground_truth, predictions)

        # True Positives (TP) is the diagonal elements of the confusion matrix
        TP = np.diag(conf_matrix)
        # False Positives (FP) is the sum of the values in the current column (excluding the diagonal) and the
        # corresponding row in the confusion matrix
        FP = np.sum(conf_matrix, axis=0) - TP
        # False Negatives (FN) is the sum of the values in the current row (excluding the diagonal) and the
        # corresponding column in the confusion matrix
        FN = np.sum(conf_matrix, axis=1) - TP
        # True Negatives (TN) is the sum of all values in the confusion matrix except TP, FP, and FN
        TN = conf_matrix.sum() - (TP + FP + FN)

        self.calculate_all_metrics(TP.sum(), TN.sum(), FP.sum(), FN.sum())

        # Normalize the confusion matrix
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        # Define class labels (replace with your class labels)
        classes = [str(i) for i in range(len(conf_matrix))]

        # Plot the Confusion Matrix
        # Create a new figure with the specified size
        plt.figure(figsize=(10, 8))

        # Set the font family and font size for the entire plot
        plt.rcParams['font.family'] = 'Palatino Linotype'
        plt.rcParams['font.size'] = 15

        # Display the confusion matrix
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)

        # Define tick marks and set x/y labels
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes, rotation=45)

        plt.xlabel('Predicted Label', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=16, fontweight='bold')

        # Add values to the Confusion Matrix plot
        thresh = conf_matrix.max() / 2.  # Threshold for color contrast
        for i, j in np.ndindex(conf_matrix.shape):
            plt.text(j, i, f'{conf_matrix[i, j]:.2f}',
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.show()

    def calculate_all_metrics(self, TP, TN, FP, FN):
        # Basic Metrics
        P = TP + FN
        N = TN + FP

        # Calculating the other metrics:
        # Sensitivity, recall, hit rate, or true positive rate
        TPR = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
        # Specificity or true negative rate
        TNR = np.divide(TN, TN + FP, out=np.zeros_like(TN, dtype=float), where=(TN + FP) != 0)
        # Precision or positive predictive value
        PPV = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
        # PPV = np.sum(TP) / np.sum(TP + FN)
        # Negative predictive value
        NPV = np.divide(TN, TN + FN, out=np.zeros_like(TN, dtype=float), where=(TN + FN) != 0)
        # Miss rate or false negative rate
        FNR = np.divide(FN, TP + FN, out=np.zeros_like(FN, dtype=float), where=(TP + FN) != 0)
        # Fall-out or false positive rate
        FPR = np.divide(FP, FP + TN, out=np.zeros_like(FP, dtype=float), where=(FP + TN) != 0)
        # False discovery rate
        FDR = np.divide(FP, TP + FP, out=np.zeros_like(FP, dtype=float), where=(TP + FP) != 0)
        # False omission rate
        FOR = np.divide(FN, FN + TN, out=np.zeros_like(FN, dtype=float), where=(FN + TN) != 0)
        # Positive likelihood ratio
        LR_PLUS = np.divide(TPR, FPR, out=np.zeros_like(TPR, dtype=float), where=FPR != 0)
        # Positive likelihood ratio
        LR_MINUS = np.divide(FNR, TNR, out=np.zeros_like(FNR, dtype=float), where=TNR != 0)
        # Prevalence threshold
        PT = (np.sqrt(TPR * (1 - TNR)) + TNR - 1) / (TPR + TNR - 1)
        # Threat score
        TS = np.divide(TP, (TP + FN + FP), out=np.zeros_like(TP, dtype=float), where=(TP + FN + FP) != 0)
        # Prevalence
        prevalence = P / (P + N)
        # Accuracy
        ACC = (TP + TN) / (P + N)
        # ACC = np.sum(TP + TN) / np.sum(TP + FP + FN + TN)
        # Balanced Accuracy (BA)
        BA = (TPR + TNR) / 2
        # F1 score: The Harmonic Mean of Precision and Sensitivity
        F1 = np.divide(2 * PPV * TPR, (PPV + TPR), out=np.zeros_like(PPV, dtype=float), where=(PPV + TPR) != 0)
        # Phi Coefficient (φ or r_φ) or Matthews Correlation Coefficient (MCC)
        MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        # Fowlkes–Mallows Index (FM)
        FM = np.sqrt(PPV * TPR)
        # Informedness or Bookmaker Informedness (BM)
        BM = TPR + TNR - 1
        # Markedness (MK) or DeltaP (Δp)
        MK = PPV + NPV - 1
        # Diagnostic Odds Ratio (DOR)
        DOR = np.divide(LR_PLUS, LR_MINUS, out=np.zeros_like(LR_PLUS, dtype=float), where=LR_MINUS != 0)

        metrics = {
            "Condition Positive (P)": P,
            "Condition Negative (N)": N,
            "True Positive (TP)": TP,
            "True Negative (TN)": TN,
            "False Positive (FP)": FP,
            "False Negative (FN)": FN,
            "Sensitivity, Recall, Hit Rate, or True Positive Rate (TPR)": TPR,
            "Specificity, Selectivity, or True Negative Rate (TNR)": TNR,
            "Precision or Positive Predictive Value (PPV)": PPV,
            "Negative Predictive Value (NPV)": NPV,
            "Miss Rate or False Negative Rate (FNR)": FNR,
            "Fall-Out or False Positive Rate (FPR)": FPR,
            "False Discovery Rate (FDR)": FDR,
            "False Omission Rate (FOR)": FOR,
            "Positive Likelihood Ratio (LR+)": LR_PLUS,
            "Negative Likelihood Ratio (LR-)": LR_MINUS,
            "Prevalence Threshold (PT)": PT,
            "Threat Score (TS) or Critical Success Index (CSI)": TS,
            "Prevalence": prevalence,
            "Accuracy (ACC)": ACC,
            "Balanced Accuracy (BA)": BA,
            "The Harmonic Mean of Precision and Sensitivity (F One Score)": F1,
            "Matthews Correlation Coefficient (MCC)": MCC,
            "Fowlkes Mallows Index (FM)": FM,
            "Informedness or Bookmaker Informedness (BM)": BM,
            "Markedness (MK)": MK,
            "Diagnostic Odds Ratio (DOR)": DOR
        }

        for metric, value in metrics.items():
            # Convert the value to list or float depending on its type
            if isinstance(value, np.ndarray):
                value = value.tolist()
            else:
                value = float(value)
            logging.info(f"{metric}: {value}")

        return metrics

    def visualize_tsne(self):
        # Initialize lists to store features and labels
        features_list = []
        labels_list = []

        for batch_idx, (inputs, labels) in enumerate(self.dataloaders['Test']):
            inputs = inputs.to(self.device, dtype=torch.float32)
            # inputs = inputs.to(self.device)

            with torch.no_grad():
                conv_features, out = self.model(inputs)
                pred = out.argmax(dim=1)

            # Append features and labels to the lists
            features_list.append(conv_features.cpu().numpy())  # You can choose which feature to use here
            labels_list.append(pred.cpu().numpy())  # Use 'pred' for predicted labels

        # Concatenate features and labels
        features = np.concatenate(features_list)
        labels = np.concatenate(labels_list)

        # Flatten the feature map of each sample to ensure it's 2D
        features = features.reshape(features.shape[0], -1)

        # Initialize t-SNE
        tsne = TSNE(n_components=2, perplexity=30)

        # Fit and transform the features
        tsne_result = tsne.fit_transform(features)

        # Create a new figure with the specified size
        plt.figure(figsize=(10, 8))

        # Set the font family and font size for the entire plot
        plt.rcParams['font.family'] = 'Palatino Linotype'
        plt.rcParams['font.size'] = 18

        # Scatter plot with t-SNE results
        # Added alpha for transparency, edgecolor for better distinction, and s for size
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap=plt.get_cmap('jet'),
                              alpha=0.6, edgecolor='w', linewidth=0.5, s=100)  # s=100 sets the size of the points

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Labels', rotation=270, labelpad=15)  # Setting the label for the colorbar

        # Ensure layout fits well
        plt.tight_layout()

        # Display the plot
        plt.show()
