# Python 3 code to rename multiple
# files in a directory or defect_path
 
# importing os module
import os
 
# Function to rename multiple files
def main():
   
    defect_path = "../pics/Defect"
    ndefect_path = "../pics/nDefect"
    for count, filename in enumerate(os.listdir(defect_path)):
        dst = f"{str(count)}.jpg"
        src =f"{defect_path}/{filename}"  # defect_pathname/filename, if .py file is outside defect_path
        dst =f"{defect_path}/{dst}"
         
        # rename() function will
        # rename all the files
        os.rename(src, dst)

    for count, filename in enumerate(os.listdir(ndefect_path)):
        dst = f"{str(count)}.jpg"
        src =f"{ndefect_path}/{filename}"  # defect_pathname/filename, if .py file is outside defect_path
        dst =f"{ndefect_path}/{dst}"
        # rename() function will
        # rename all the files
        os.rename(src, dst)
 
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()