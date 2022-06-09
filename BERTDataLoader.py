import torch
import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {'translate':0,
          'transfer':1,
          'timer':2,
          'definition':3,
          'meaning_of_life':4,
          'insurance_change':5,
          'find_phone':6,
          'travel_alert':7,
          'pto_request':8,
          'improve_credit_score':9,
          'fun_fact':10, 
          'change_language':11,
          'payday':12,
          'replacement_card_duration':13,
          'time':14,
          'application_status':15,
          'flight_status':16,
          'flip_coin':17,
          'change_user_name':18, 
          'where_are_you_from':19,
          'shopping_list_update':20,
          'what_can_i_ask_you':21,
          'maybe':22,
          'oil_change_how':23,
          'restaurant_reservation':24,
          'balance':25,
          'confirm_reservation':26,
          'freeze_account':27,
          'rollover_401k':28,
          'who_made_you':29,
          'distance':30,
          'user_name':31,
          'timezone':32,
          'next_song':33,
          'transactions':34,
          'restaurant_suggestion':35,
          'rewards_balance':36,
          'pay_bill':37,
          'spending_history':38,
          'pto_request_status':39,
          'credit_score':40,
          'new_card':41,
          'lost_luggage':42,
          'repeat':43,
          'mpg':44,
          'oil_change_when':45,
          'yes':46,
          'travel_suggestion':47,
          'insurance':48,
          'todo_list_update':49,
          'reminder':50,
          'change_speed':51,
          'tire_pressure':52,
          'no':53,
          'apr':54,
          'nutrition_info':55,
          'calendar':56,
          'uber':57,
          'calculator':58,
          'date':59,
          'carry_on':60,
          'pto_used':61,
          'schedule_maintenance':62,
          'travel_notification':63,
          'sync_device':64,
          'thank_you':65,
          'roll_dice':66,
          'food_last':67,
          'cook_time':68,
          'reminder_update':69,
          'report_lost_card':70,
          'ingredient_substitution':71,
          'make_call':72,
          'alarm':73,
          'todo_list':74,
          'change_accent':75,
          'w2':76,
          'bill_due':77,
          'calories':78,
          'damaged_card':79,
          'restaurant_reviews':80,
          'routing':81,
          'do_you_have_pets':82,
          'schedule_meeting':83,
          'gas_type':84,
          'plug_type':85,
          'tire_change':86,
          'exchange_rate':87,
          'next_holiday':88,
          'change_volume':89,
          'who_do_you_work_for':90,
          'credit_limit':91,
          'how_busy':92,
          'accept_reservations':93,
          'order_status':94,
          'pin_change':95,
          'goodbye':96,
          'account_blocked':97,
          'what_song':98,
          'international_fees':99,
          'last_maintenance':100,
          'meeting_schedule':101,
          'ingredients_list':102,
          'report_fraud':103,
          'measurement_conversion':104,
          'smart_home':105,
          'book_hotel':106,
          'current_location':107,
          'weather':108,
          'taxes':109,
          'min_payment':110,
          'whisper_mode':111,
          'cancel':112,
          'international_visa':113,
          'vaccines':114,
          'pto_balance':115,
          'directions':116,
          'spelling':117,
          'greeting':118,
          'reset_settings':119,
          'what_is_your_name':120,
          'direct_deposit':121,
          'interest_rate':122,
          'credit_limit_change':123,
          'what_are_your_hobbies':124,
          'book_flight':125,
          'shopping_list':126,
          'text':128,
          'bill_balance':129,
          'share_location':130,
          'redeem_rewards':131,
          'play_music':132,
          'calendar_update':133,
          'are_you_a_bot':134,
          'gas':135,
          'expiration_date':136,
          'update_playlist':137,
          'cancel_reservation':138,
          'tell_joke':139,
          'change_ai_name':140,
          'how_old_are_you':141,
          'car_rental':142,
          'jump_start':143,
          'meal_suggestion':144,
          'recipe':145,
          'income':146,
          'order':147,
          'traffic':148,
          'order_checks':149,
          'card_declined':150
          }

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        f = open(path, "r")
        content = f.read()
        phraseList = content.split("\n")
        #self.labels = np.empty(len(phraseList), dtype=np.float64)
        curTexts = []
        curLabels = []
        for phrase in phraseList:
          p = phrase.split("\t")
          #np.append(self.labels,p[0])
          curTexts.append(p[1])
          curLabels.append(p[0])
        ## TODO: get this back!!!!self.labels = Variable(th.from_numpy(self.labels)).type(torch.LongTensor)
        #TODO: comment this back later
        #self.labels = torch.as_tensor(self.labels, dtype=torch.long, device=torch.device("cpu"))
        #TODO: comment above back
        #self.labels = torch.Tensor(self.labels)
        self.labels = [int(labels[label]) for label in curLabels]
        print("here, worked before")
        print(path)
        #self.texts = [tokenizer(text, padding='max_length', max_length = 512, truncation = True,
                                #return_tensors="pt") for text in curTexts]
        # self.texts = [tokenizer(text, padding=True, truncation=True,
        #                         return_tensors="pt") for text in curTexts]
        self.lens = [len(text) for text in curTexts]
        #TODO: change here, change the padding
        #, 
        #                       padding='max_length', max_length = 512
        #TODO: comment below back
        self.texts = [tokenizer(text, truncation=True,
                                return_tensors="pt") for text in curTexts]
        #TODO: comment above back
        # print("text print")
        # print(len(self.texts))
        # print(self.texts[0])
        # print(len(self.lens))
        # print(self.lens[0])
        # print(len(self.labels))
        # print(self.labels[0])
        # print("ends here")
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        #return torch.from_numpy(np.array(self.labels[idx]))
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def get_batch_lens(self, idx): 
        return self.lens[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        lens = self.get_batch_lens(idx)
        return [batch_texts, batch_y, lens]
        #return [batch_texts, batch_y]


