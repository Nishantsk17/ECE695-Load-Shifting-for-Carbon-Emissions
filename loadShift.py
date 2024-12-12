import torch
import torch.nn as nn
from dataLoader_2 import carbon_data, power_data # Import the right dataloader file here
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define the policy network
class WorkloadShiftPolicy(nn.Module):
    def __init__(self, input_dim):
        super(WorkloadShiftPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

# Function to calculate emissions based on shift percentage
def calculate_emissions(day1_actual, day2_actual, power):
    # Calculate emissions for both days using the actual carbon intensity trace
    emissions_day1 = power * day1_actual
    emissions_day2 = power * day2_actual
    return emissions_day1.sum(), emissions_day2.sum()

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, stage, confidence):
        super().__init__()
        self.stage = stage
        self.carbon_trace, self.interval_trace, self.point_trace = carbon_data(stage, confidence)
        self.power_trace = power_data(stage)

        #print(self.interval_trace.shape)
        # Calculate mean and std for interval_trace only
        flattened_interval = self.interval_trace.reshape(-1, self.interval_trace.shape[-1])
        self.interval_mean = flattened_interval.mean(axis=0)
        self.interval_std = flattened_interval.std(axis=0)

    def __len__(self):
        return len(self.carbon_trace)-1

    def __getitem__(self, idx):
        # Get interval and power trace for the given index
        interval_data_1 = self.interval_trace[idx]
        interval_data_2 = self.interval_trace[idx + 1]

        # Normalize interval data
        interval_data_1 = (interval_data_1 - self.interval_mean) / self.interval_std
        interval_data_2 = (interval_data_2 - self.interval_mean) / self.interval_std

        # Flatten and concatenate interval data
        interval_data = np.concatenate((interval_data_1.flatten(), interval_data_2.flatten()))

        power_data = self.power_trace[idx]  # Shape: (24,)
        ground_truth_carbon_1 = self.carbon_trace[idx]
        ground_truth_carbon_2 = self.carbon_trace[idx + 1]
        pred_carbon_1 = self.point_trace[idx]
        pred_carbon_2 = self.point_trace[idx+1]

        # Convert to tensor
        input_data = np.concatenate((interval_data, power_data))
        input_data = torch.tensor(input_data, dtype=torch.float32)
        # ground_truth_carbon_1 = torch.tensor(ground_truth_carbon_1, dtype=torch.float32)
        # ground_truth_carbon_2 = torch.tensor(ground_truth_carbon_2, dtype=torch.float32)
        em1, em2 = calculate_emissions(ground_truth_carbon_1, ground_truth_carbon_2, power_data)
        label = 0 if em1 <= em2 else 1
        label = torch.tensor(label, dtype=torch.float32)
    
        em1_pred, em2_pred = calculate_emissions(pred_carbon_1, pred_carbon_2, power_data)
        carbon_cast_label = 0 if em1_pred <= em2_pred else 1
        pred_em = em1_pred if carbon_cast_label == 0 else em2_pred
        carbon_cast_label = torch.tensor(carbon_cast_label, dtype=torch.float32)

        if self.stage == "train":
            return input_data, label

        elif self.stage == "test":
            return input_data, label, carbon_cast_label, em1, em2, pred_em
    

def train(model, confidence):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(device)
    model = model.to(device)
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.99))
    epochs = 650
    my_dataset = MyDataset("train", confidence)
    print(len(my_dataset))
    train_dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=32, shuffle=False, num_workers=4)

    loss_data = []
    for epoch in range(epochs):
        running_loss = 0.0
        b = 0
        for i, data in enumerate(train_dataloader):
            inputs,label = data
            inputs = inputs.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            b += 1

        avg_loss = running_loss / b
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss}")
            loss_data.append(avg_loss)
    
    torch.save(model.state_dict(), f"model_{confidence}")
    return loss_data

def test(model, state_path, confidence):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(state_path, map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode

    total_samples = 0
    correct_predictions = 0
    carbon_cast_pred = 0
    all_labels = []
    all_outputs = []

    criterion = torch.nn.BCELoss(reduction='mean')  # Same loss as in training
    total_loss = 0.0
    my_dataset = MyDataset("test", confidence)
    print(len(my_dataset))
    test_loader = torch.utils.data.DataLoader(my_dataset, batch_size=1, shuffle=False)
    carbon_cast_em = 0  # Carbon-cast emissions
    total_em = 0  # Total emissions based on model decisions
    no_shift_em = 0
    with torch.no_grad():  # Disable gradient computation
        for i, data in enumerate(test_loader):
            inputs,labels, pred_labels, em1, em2, pred_em = data
            pred_labels = pred_labels.to(device)
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            # Forward pass
            outputs = model(inputs).squeeze(dim=-1)

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Convert probabilities to binary predictions
            predictions = (outputs >= 0.5).float()  # Threshold = 0.5

            # Update total_em based on model's prediction
            for idx, prediction in enumerate(predictions):
                if prediction == 0:
                    total_em += em1[idx].item()  # Add em1 for label 0
                else:
                    total_em += em2[idx].item()  # Add em2 for label 1
            
            # Update carbon_cast_em based on pred_em
            carbon_cast_em += pred_em.sum().item()  # Add pred_em for all samples in the batch
            no_shift_em += em1.sum().item()

            # Collect predictions and labels for metrics
            all_outputs.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Count correct predictions
            correct_predictions += (predictions == labels).sum().item()
            carbon_cast_pred += (pred_labels == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate accuracy
    accuracy = correct_predictions / total_samples
    accuracy1 = carbon_cast_pred / total_samples
    conf_matrix = confusion_matrix(all_labels, all_outputs)
    
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Carbon-Cast Accuracy: {accuracy1 * 100:.2f}%")
    print(f"Total Emissions (Model Decisions): {total_em}")
    print(f"Total Emissions (Carbon Cast): {carbon_cast_em}")
    print(f"Emissions without any shifting: {no_shift_em}")
    print(f"Emissions saved: {carbon_cast_em - total_em}")
    print(f"Save%: {((carbon_cast_em - total_em)/carbon_cast_em)*100}%")

    # Data for the bar graph
    categories = [
        "Model",
        "Carbon Cast",
    ]
    values = [total_em, carbon_cast_em]

    # Create the bar graph
    plt.figure(figsize=(8, 5))
    x_positions = np.arange(len(categories))  # Generate evenly spaced x positions
    plt.bar(x_positions, values, width=0.4)
    plt.ylim(434000, 435000)
    # Adding titles and labels
    plt.title("Emissions Comparison")
    plt.ylabel("Emissions (gCO2)")
    plt.xlabel("Scenario")

    # Rotating category labels for better readability
    plt.xticks(x_positions, categories, rotation=30, ha="right")

    # Display the graph
    plt.tight_layout()
    plt.savefig(f"Emissions_{confidence}.png")
    plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Don't shift", "Shift"])
    disp.plot(cmap=plt.cm.Blues)

    plt.title("Confusion Matrix")
    plt.savefig(f"confusion_matrix_{confidence}.png")  # Save the figure as an image
    plt.show()
    return

if __name__ == "__main__":

    input_dim = 120  # 24-hour predicted intensity + power for two days
    model = WorkloadShiftPolicy(input_dim)
    confidence = 0.05 # Confidence for carbon intensity interval data

    # Train the policy
    # loss = train(model, confidence)
    # plt.figure()
    # plt.plot(loss)
    # plt.savefig(f"loss_plot2_{confidence}.png")
    # #plt.show()

    # Test the policy
    # test(model, f"model2_{confidence}", confidence)
