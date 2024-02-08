import pandas as pd
import joblib

# Input for CSV file names
events_data_file = input("Enter the path for the events data CSV file: ")
subscribers_data_file = input("Enter the path for the subscribers data CSV file: ")

# Maximum number of predictions to display, tunable parameter
max_preds = 10

# Load the events data, ensuring no duplicates
events_data_new = pd.read_csv(events_data_file)
events_data_new = events_data_new.drop_duplicates()

# Load the subscribers data, ensuring no duplicates
subscribers_data = pd.read_csv(subscribers_data_file)
subscribers_data = subscribers_data.drop_duplicates()

# Load the predictive model
model = joblib.load('Model.pkl')

# Ensure member_id data type is string for consistency
events_data_new['member_id'] = events_data_new['member_id'].astype('str')

# Convert event_type to one-hot encoding and drop unnecessary columns
event_type_OHE = pd.get_dummies(events_data_new['event_type'])
events_data_new = events_data_new.drop(["event_type", "dt"], axis=1)
events_data_new[event_type_OHE.columns] = event_type_OHE

# Group data by member_id and summarize
events_data_new_gb = events_data_new.groupby('member_id').sum().reset_index()

# Determine if a sale happened based on pt_sale values
events_data_new_gb['pt_sale_category'] = events_data_new_gb['pt_sale'].apply(lambda x: 1 if x > 0 else 0)

# Calculate the ratio of chat messages to SMS sent, handling division by zero
events_data_new_gb["chat_sms_ratio"] = events_data_new_gb["chat_message_sent"] / (events_data_new_gb["sms_sent"] + events_data_new_gb["chat_message_sent"])
events_data_new_gb["chat_sms_ratio"] = events_data_new_gb["chat_sms_ratio"].fillna(0)

# Columns to be used for prediction
cols = ['app_interaction', 'automated_email_sent', 'fitness_consultation',
        'human_communication', 'manual_email_sent',
        'personal_appointment_scheduled', 'pt_usage', 'usage',
        'chat_sms_ratio']

# Make predictions using the model
predictions = model.predict(events_data_new_gb[cols])
events_data_new_gb["pt_sale_pred"] = predictions

# Merge predictions with subscriber data
merged_df = pd.merge(events_data_new_gb, subscribers_data[['member_id', 'segment_code']], on='member_id', how='left')

# Filter for predicted sales
filtered_df = merged_df[merged_df['pt_sale_pred'] == 1]

# Group by segment code and limit the number of member IDs to max_preds
grouped_df = filtered_df.groupby('segment_code')['member_id'].apply(list).reset_index()
grouped_df['member_id'] = grouped_df['member_id'].apply(lambda x: x[:max_preds])

# Count the total number of pt_sale members per segment
grouped_df['total pt_sale members'] = grouped_df['member_id'].apply(len)

grouped_df.to_csv("total pt_sale members.csv", index=False)

print("total pt_sale members.csv has been downloaded")