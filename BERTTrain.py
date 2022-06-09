from torch.optim import Adam
from tqdm import tqdm
from BERTDataLoader import *
from BERTModel import *
from transformers import DataCollatorWithPadding
import math
import torch.nn.functional as F

def collate_fn(batch):
    print("hello")
    for dat in batch: 
        print(dat)
    """
    args:
        batch: [[input_vector, label_vector] for seq in batch]
    return:
        [[output_vector]] * batch_size, [[label]]*batch_szie
    """
    percentile = 100
    dynamical_pad = True
    max_len = 50
    pad_index = 0
 
    print("1")
    #lens = [len(dat[0]) for dat in batch]
    lens = [dat[2] for dat in batch]
    # find the max len in each batch
    # if dynamical_pad:
    #     # dynamical padding
    #     seq_len = min(math.floor(np.percentile(lens, percentile)), max_len)
    #     #or seq_len = max(lens)
    # else:
    #     # fixed length padding
    #     seq_len = max_len

    return_bat = []
    for dat in batch:
        #issue here
        # print("This is printed here")
        # print(dat)
        # print("This is printed here")

        #what to pad
        seq = dat[0]['input_ids']
        seq1 = dat[0]['token_type_ids']
        seq2 = dat[0]['attention_mask']

        #edit here, pad tensor to given length
        paddingList = [dat[2] for dat in batch]
        padding = max(paddingList) - dat[2]
        pad_ten = torch.zeros(1,padding)
        nseq = torch.cat((seq,pad_ten),1)
        nseq1 = torch.cat((seq1,pad_ten),1)
        nseq2 = torch.cat((seq2,pad_ten),1)
        dat[0]['input_ids'] = seq
        dat[0]['token_type_ids'] = seq1
        dat[0]['attention_mask'] = seq2
        return_bat.append(dat)

    #check the format here
    texts = [dat[0] for dat in return_bat]
    labels = [dat[1] for dat in return_bat]
    labels = np.array(labels)
    print("texts")
    print(texts[0])
    return texts, labels
 
def train(model, train_data, val_data, learning_rate, epochs):
    train = Dataset(train_data)
    val = Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True, collate_fn = collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2, shuffle=True, collate_fn = collate_fn)
    #train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    #val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #TODO: check this, crossentrypy loss
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):
                #change start
                print("hihihihihihih")
                print(train_input)
                train_input = train_input[0]
                train_label = train_label[0]
                print(train_input)
                print(train_label)
                #TODO: change here, the only problem is here
                train_label = torch.tensor([train_label],dtype=torch.long)
                print(train_label)
                print("yesyesyesyes")
                #change end

                train_label = train_label.to(device)
                print("attention_mask")
                print(train_input['attention_mask'])
                mask = train_input['attention_mask'].to(device)
                print(mask)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                print("output: ")
                print(output)
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:
                    val_input = val_input[0]
                    val_label = val_label[0]
                    #TODO: check heres, the only problem is here
                    val_label = torch.tensor([train_label],dtype=torch.long)
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')
                  
EPOCHS = 5
model = BertClassifier()
LR = 1e-6
              
train(model, "/Users/wangyilin/Desktop/yilinWang\'sClassifier/ind_try", "/Users/wangyilin/Desktop/yilinWang\'sClassifier/ind_try", LR, EPOCHS)