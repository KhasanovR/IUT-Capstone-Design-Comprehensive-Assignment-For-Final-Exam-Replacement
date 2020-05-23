import cv2
import numpy as np




filename = "input_1280-720.mp4"

cap = cv2.VideoCapture(filename)



width = 400
height = 300
size = (width, height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("input_"+str(width)+"-"+str(height)+".mp4", fourcc, 30, size) 

# Source points for homography
src = np.array([
    ((width*2)/10, height), 
    ((width*9)/10, height),
    ((width*4)/7, (7*height)/10), 
    ((width*5)/10, (7*height)/10)], dtype="float32")




while cap.isOpened():
    
    ret, frame = cap.read()
    if ret is False:
        break
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    

    out.write(frame)

    q = cv2.waitKey(1) & 0xff

    if q == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




