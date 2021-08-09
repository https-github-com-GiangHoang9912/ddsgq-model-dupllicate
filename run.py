from factory import Factory
from model.model import Bert_Model, SBert, TBert, LayerNorm, Output_Layer
import os

class OutputStyler:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    CYAN = '\033[36m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


DASH = '='
user_input = OutputStyler.CYAN + ':-$ ' + OutputStyler.ENDC

def headerize(text=DASH):

    result = os.get_terminal_size()

    terminal_length = result.columns

    if text:
        text_length = len(text)
        remaining_places = int(terminal_length) - text_length
        if remaining_places > 0:
            return DASH * (remaining_places // 2 - 1) + ' ' + text + ' ' + DASH * (remaining_places // 2 - 1)

    else:

        return DASH * int(terminal_length) 

def show(inpt,fit,top,confident):
    os.system("cls")
    print(headerize("Question"))
    if inpt != "":
        print("-## "+inpt)
    print(headerize("Duplicate sentence - confident: "+str(confident)))
    for k,v in fit.items():
        print("-# "+k+" - "+str(v))
    print(headerize("Top 5"))
    for idx,(k,v) in enumerate(top.items()):
        print("-# "+k+" - "+str(v))
        if idx == 4:
            break
    print(headerize("Input sentence"))

f = Factory()


fit = {}
top = {}
confident = 0.6
inpt = ""
while True:
    show(inpt,fit,top,confident)
    inpt = input(user_input)
    fit,top = f.find_duplicate([inpt],confident)


    


             
            

    

