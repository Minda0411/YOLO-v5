#import os,shutil

# path='runs'
## Idea: in the given dir, find the newest foler by time
## replace
def copy_weight(search_path="./runs", filename = "weights/best.pt", new_path = "./weights/best.pt"):
    import os,shutil
    newest=max([os.path.join(search_path, d) for d in os.listdir(search_path)],key=os.path.getmtime)
    weight_path=r'weights/best.pt'
    weight_path=os.path.join(newest,weight_path)
    dst_path=r'./weights/best.pt'
    if os.path.exists:
        shutil.copyfile(weight_path,dst_path)  
    print("File is updated...")  
    return

def main():
    path = 'runs'
    copy_weight(search_path=path, filename = "weights/best.pt", new_path = "./weights/best.pt")

if __name__ == "__main__":
    main()
# newest=max([os.path.join(path,d) for d in os.listdir(path)],key=os.path.getmtime)
# weight_path=r'weights/best.pt'
# weight_path=os.path.join(newest,weight_path)
# dst_path=r'weights/best.pt'
# if os.path.exists:
#     shutil.copyfile(weight_path,dst_path)  
# print("copy")  