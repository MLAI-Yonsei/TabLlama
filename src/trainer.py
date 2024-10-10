import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

class Trainer:
    def __init__(
        self,
        args,
        model,
        train_loader,
        val_loader,
        test_loader,
        num_epochs,
        device,
        num_classes,
        lr,
        patience=5,
        min_delta=0.001,
    ):
        self.args = args
        self.model = model
        self.train_dataloader = train_loader
        self.eval_dataloader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.device = device
        self.num_classes = num_classes
        self.lr = lr
        self.patience = patience
        self.min_delta = min_delta
        
    def train(self):
        self.model.to(self.device)
        criterion1 = nn.MSELoss()
        criterion2 = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        best_loss = float("inf")
        best_model = None
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for batch in self.train_dataloader:
                inputs, targets, masks = (
                    batch["data"],
                    batch["intent_label"],
                    batch["padding_mask"],
                )

                inputs, targets, masks = (
                    inputs.long().to(self.device),
                    targets.long().to(self.device),
                    masks.to(self.device),
                )

                optimizer.zero_grad()

                output_seq, intent = self.model(inputs[:, :-1, :], masks[:, :-1])

                loss1 = criterion1(output_seq, inputs[:, -1, :].float())
                loss2 = criterion2(intent, targets)

                loss = loss1 + loss2

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dataloader)

            # Evaluation
            eval_results = self.evaluate(self.eval_dataloader)
            

            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}, "
                f"Val Loss: {eval_results['avg_val_loss']:.4f}, MAE: {eval_results['mae']:.4f}, RMSE: {eval_results['rmse']:.4f}, "
                f"Non-zero MAE: {eval_results['non_zero_mae']:.4f}, Non-zero RMSE: {eval_results['non_zero_rmse']:.4f}, "
                f"Intent Accuracy: {eval_results['intent_accuracy']:.4f}"
            )

            # Early stopping check
            if eval_results["avg_val_loss"] < best_loss - self.min_delta:
                best_loss = eval_results["avg_val_loss"]
                best_model = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # save best model
        if best_model is not None:
            self.model.load_state_dict(best_model)
            torch.save(best_model, self.args.save_path)
            print(f"Best model saved as '{self.args.save_path}'")

        # 테스트 세트에 대한 평가
        test_results = self.evaluate(self.test_loader)

        print("Test Results:")
        for key, value in test_results.items():
            print(f"{key}: {value:.4f}")

        return self.model

    def evaluate(self, data_loader):
        self.model.eval()
        true_seq_values = []
        predicted_seq_values = []
        true_intent_values = []
        predicted_intent_values = []
        total_val_loss = 0

        criterion1 = nn.MSELoss()
        criterion2 = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in data_loader:
                inputs, targets, masks = (
                    batch["data"],
                    batch["intent_label"],
                    batch["padding_mask"],
                )

                inputs, targets, masks = (
                    inputs.long().to(self.device),
                    targets.long().to(self.device),
                    masks.to(self.device),
                )

                output_seq, intent = self.model(inputs[:, :-1, :], masks[:, :-1])

                # Calculate validation loss
                loss1 = criterion1(output_seq, inputs[:, -1, :].float())
                loss2 = criterion2(intent, targets)

                val_loss = loss1 + loss2
                total_val_loss += val_loss.item()
                
                output_seq = torch.round(output_seq).long()

                # Sequence output processing
                true_seq = inputs[:, -1, :].float().cpu().numpy()
                predicted_seq = output_seq.cpu().numpy()
                
                # Flatten the 2D arrays to 1D for metric calculations
                true_seq_values.extend(true_seq.flatten())
                predicted_seq_values.extend(predicted_seq.flatten())

                # Intent output processing (multi-class classification)
                true_intent = targets.cpu().numpy()
                predicted_intent = intent.argmax(dim=1).cpu().numpy()

                true_intent_values.extend(true_intent)
                predicted_intent_values.extend(predicted_intent)

        # Convert lists to numpy arrays
        true_seq_values = np.array(true_seq_values)
        predicted_seq_values = np.array(predicted_seq_values)

        # Sequence evaluation metrics
        mae = mean_absolute_error(true_seq_values, predicted_seq_values)
        rmse = np.sqrt(mean_squared_error(true_seq_values, predicted_seq_values))

        # Calculate Non-zero MAE and RMSE
        non_zero_mask = true_seq_values != 0
        if non_zero_mask.sum() == 0:
            non_zero_mae = 999
            non_zero_rmse = 999
        else:
            non_zero_true = true_seq_values[non_zero_mask]
            non_zero_pred = predicted_seq_values[non_zero_mask]
            non_zero_mae = mean_absolute_error(non_zero_true, non_zero_pred)
            non_zero_rmse = np.sqrt(mean_squared_error(non_zero_true, non_zero_pred))

        # Calculate accuracy with rounded sequence values
        rounded_true_seq = np.round(true_seq_values)
        rounded_predicted_seq = np.round(predicted_seq_values).clip(min=0)

        seq_accuracy = accuracy_score(rounded_true_seq, rounded_predicted_seq)

        # Calculate Non-zero accuracy (sequence)
        non_zero_mask = rounded_true_seq != 0
        
        if non_zero_mask.sum() == 0:
            non_zero_acc = 0
        else:
            non_zero_acc = accuracy_score(
                rounded_true_seq[non_zero_mask], rounded_predicted_seq[non_zero_mask]
            )

        # Calculate Intent accuracy
        intent_accuracy = accuracy_score(true_intent_values, predicted_intent_values)

        # Calculate average validation loss
        avg_val_loss = total_val_loss / len(data_loader)

        results = {
            "mae": mae,
            "rmse": rmse,
            "non_zero_mae": non_zero_mae,
            "non_zero_rmse": non_zero_rmse,
            "intent_accuracy": intent_accuracy,
            "avg_val_loss": avg_val_loss
        }

        return results

    def test_and_predict(self, test_loader):
        self.model.eval()
        true_seq_values = []
        predicted_seq_values = []
        true_intent_values = []
        predicted_intent_values = []
        total_test_loss = 0

        criterion1 = nn.MSELoss()
        criterion2 = nn.CrossEntropyLoss()

        all_predictions = []
        all_true_values = []
        all_intents = []
        all_predicted_intents = []

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets, masks = batch["data"], batch["intent_label"], batch["padding_mask"]
                inputs, targets, masks = inputs.long().to(self.device), targets.long().to(self.device), masks.to(self.device)

                output_seq, intent = self.model(inputs[:, :-1, :], masks[:, :-1])

                # Calculate test loss
                loss1 = criterion1(output_seq, inputs[:, -1, :].float())
                loss2 = criterion2(intent, targets)
                test_loss = loss1 + loss2
                total_test_loss += test_loss.item()

                output_seq = torch.round(output_seq).long()

                # Sequence output processing
                true_seq = inputs[:, -1, :].float().cpu().numpy()
                predicted_seq = output_seq.cpu().numpy()
                
                # Store predictions and true values
                all_predictions.extend(predicted_seq)
                all_true_values.extend(true_seq)
                all_intents.extend(targets.cpu().numpy())

                # Flatten the 2D arrays to 1D for metric calculations
                true_seq_values.extend(true_seq.flatten())
                predicted_seq_values.extend(predicted_seq.flatten())

                # Intent output processing (multi-class classification)
                true_intent = targets.cpu().numpy()
                predicted_intent = intent.argmax(dim=1).cpu().numpy()

                true_intent_values.extend(true_intent)
                predicted_intent_values.extend(predicted_intent)
                all_predicted_intents.extend(predicted_intent)

        # Convert lists to numpy arrays
        true_seq_values = np.array(true_seq_values)
        predicted_seq_values = np.array(predicted_seq_values)

        # Overall metrics calculation
        mae = mean_absolute_error(true_seq_values, predicted_seq_values)
        rmse = np.sqrt(mean_squared_error(true_seq_values, predicted_seq_values))

        # Calculate Non-zero MAE and RMSE
        non_zero_mask = true_seq_values != 0
        if non_zero_mask.sum() > 0:
            non_zero_true = true_seq_values[non_zero_mask]
            non_zero_pred = predicted_seq_values[non_zero_mask]
            non_zero_mae = mean_absolute_error(non_zero_true, non_zero_pred)
            non_zero_rmse = np.sqrt(mean_squared_error(non_zero_true, non_zero_pred))
        else:
            non_zero_mae = non_zero_rmse = float('nan')

        # Calculate accuracy with rounded sequence values
        rounded_true_seq = np.round(true_seq_values)
        rounded_predicted_seq = np.round(predicted_seq_values).clip(min=0)
        seq_accuracy = accuracy_score(rounded_true_seq, rounded_predicted_seq)

        # Calculate Non-zero accuracy (sequence)
        non_zero_mask = rounded_true_seq != 0
        if non_zero_mask.sum() > 0:
            non_zero_acc = accuracy_score(rounded_true_seq[non_zero_mask], rounded_predicted_seq[non_zero_mask])
        else:
            non_zero_acc = float('nan')

        # Calculate Intent accuracy
        intent_accuracy = accuracy_score(true_intent_values, predicted_intent_values)

        # Calculate average test loss
        avg_test_loss = total_test_loss / len(self.test_loader)

        results = {
            "mae": mae,
            "rmse": rmse,
            "non_zero_mae": non_zero_mae,
            "non_zero_rmse": non_zero_rmse,
            "intent_accuracy": intent_accuracy,
            "avg_test_loss": avg_test_loss
        }

        intent_metrics = {}
        unique_intents = np.unique(all_intents)

        intent_to_index = {intent: idx for idx, intent in enumerate(unique_intents)}

        precision, recall, f1_score, support = precision_recall_fscore_support(
            all_intents, all_predicted_intents, labels=unique_intents, average=None
        )

        for intent in unique_intents:
            idx = intent_to_index[intent]
            mask = np.array(all_intents) == intent
            true = np.array(all_true_values)[mask].flatten()
            pred = np.array(all_predictions)[mask].flatten()
            
            intent_mae = mean_absolute_error(true, pred)
            intent_rmse = np.sqrt(mean_squared_error(true, pred))
            
            non_zero_mask = true != 0
            if non_zero_mask.sum() > 0:
                non_zero_true = true[non_zero_mask]
                non_zero_pred = pred[non_zero_mask]
                intent_non_zero_mae = mean_absolute_error(non_zero_true, non_zero_pred)
                intent_non_zero_rmse = np.sqrt(mean_squared_error(non_zero_true, non_zero_pred))
            else:
                intent_non_zero_mae = intent_non_zero_rmse = float('nan')
            
            rounded_true = np.round(true)
            rounded_pred = np.round(pred).clip(min=0)
            intent_seq_accuracy = accuracy_score(rounded_true, rounded_pred)
            
            if non_zero_mask.sum() > 0:
                intent_non_zero_acc = accuracy_score(rounded_true[non_zero_mask], rounded_pred[non_zero_mask])
            else:
                intent_non_zero_acc = float('nan')
            
            # Calculate intent-specific accuracy
            intent_true = np.array(all_intents) == intent
            intent_pred = np.array(all_predicted_intents) == intent
            intent_accuracy = accuracy_score(intent_true, intent_pred)
            
            intent_metrics[f"intent_{intent}"] = {
                "mae": intent_mae,
                "rmse": intent_rmse,
                "non_zero_mae": intent_non_zero_mae,
                "non_zero_rmse": intent_non_zero_rmse,
                "precision": precision[idx],
                "recall": recall[idx],
                "f1_score": f1_score[idx],
                "support": support[idx]
            }

        intent_classification_metrics = precision_recall_fscore_support(all_intents, all_predicted_intents, average=None)
        
        for i, intent in enumerate(unique_intents):
            intent_metrics[f"intent_{intent}"].update({
                "precision": intent_classification_metrics[0][i],
                "recall": intent_classification_metrics[1][i],
                "f1_score": intent_classification_metrics[2][i],
                "support": intent_classification_metrics[3][i]
            })

        results["intent_metrics"] = intent_metrics


        return results, all_predictions
