
import glob
from generator_utils import *
import real_symbol_utils


def rotateBoundingBoxes(boundingBoxesToRemove,dim):
    height=dim[0]
    width=dim[0]
    newBoundingBoxes=[]
    for point1,point2 in boundingBoxesToRemove:
        x,y=point1
        x1,y1=point2
        newBoundingBoxes.append([[height-y1,x],[height-y,x1]])
    return newBoundingBoxes

def normalizePoints(boundingBoxesToRemove,dim,newDim):
    newPoints=[]
    imageW=dim[0]
    imageH=dim[1]
    newImageW=newDim[0]
    newImageH=newDim[1]
    for point1,point2 in boundingBoxesToRemove:
        x,y=point1
        x1,y1=point2
        x,y,x1,y1=x/imageW,y/imageH,x1/imageW,y1/imageH
        x,y,x1,y1=round(x*newImageW),round(y*newImageH),round(x1*newImageW),round(y1*newImageH)
        newPoints.append([[x,y],[x1,y1]])
    return newPoints



def generate_image_with_real_background(boundingBoxesToRemove,real_symbols_ratio,sample_real,sample_real_Clean,sample,canvas,
                   dim,
                   generator_dim=(4624,3468),
                   real_symbols_in_real_backgrounds=False):
    # canvas = np.full(dim, 255) #Size of final image
    canvas=canvas.copy()

    # check if the background is portrait 
    if(dim[0]/dim[1]>1.0):
        #rotate background
        canvas=cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
        boundingBoxesToRemove=rotateBoundingBoxes(boundingBoxesToRemove,dim)
        
    # check if background resolution is bigger than generator res 
    if(dim[0]/dim[1]>generator_dim[1]/generator_dim[0]):
        widthRatio=generator_dim[1]/dim[0]
        heightRatio=generator_dim[0]/dim[1]
        if(widthRatio<heightRatio):
            canvas = cv2.resize(canvas, (int(round(canvas.shape[1]*widthRatio)),int(round(canvas.shape[0]*widthRatio))))
        else:
            canvas = cv2.resize(canvas, (int(round(canvas.shape[1]*heightRatio)),int(round(canvas.shape[0]*heightRatio))))
        newDim=canvas.shape
        dim=newDim
        boundingBoxesToRemove=normalizePoints(boundingBoxesToRemove,dim,newDim)
    
    
        
    # add padding to the smaller side of the image
    if(generator_dim[1]>dim[1] or generator_dim[0]>dim[0]):
        paddingSizeWidth=generator_dim[1]-dim[1]
        paddingSizeHeight=generator_dim[0]-dim[0]
        # divide padding to both sides
        paddingToSideWitdh=paddingSizeWidth//2
        paddingToSideHeight=paddingSizeHeight//2
        canvas=cv2.copyMakeBorder(canvas, paddingToSideHeight, paddingSizeHeight-paddingToSideHeight, paddingToSideWitdh, paddingSizeWidth-paddingToSideWitdh, cv2.BORDER_CONSTANT,value=(255,255,255))

        newBoundingBoxPoints=[]
        for point1,point2 in boundingBoxesToRemove:
            x,y=point1
            x1,y1=point2
            newBoundingBoxPoints.append([[x+paddingToSideWitdh,y+paddingToSideHeight],[x1+paddingToSideWitdh,y1+paddingToSideHeight]])
        boundingBoxesToRemove=newBoundingBoxPoints

    
    location_placement = []
    locations = [] #Store location of the symbols
    locations_units = []
    
    labels = [] #Store labels
    labels_units = []
    
    rotations = []
    

    sample_labels = list(sample.keys())

    for symbolRange in boundingBoxesToRemove: # Nr of symbols on image
        label = sample_labels[randint(0,len(sample_labels)-1)]
        startRangePoint,endRangePoint=symbolRange
        dim=(endRangePoint[0]-startRangePoint[0],endRangePoint[1]-startRangePoint[1])
        img_scale = random.uniform(0.8,1.2)
        img = get_random(label, sample)
        img = resize_by_scale(img, img_scale)

        if random.uniform(0, 1) > real_symbols_ratio or not real_symbols_in_real_backgrounds:
                img = get_random(label, sample)
                from_real_film = False
                img = resize_by_scale(img, img_scale*0.8)
        else:
            if(label in sample_real_Clean):
                #There might not be sample from real film
                imgClean,imgDirty = real_symbol_utils.get_random_pair(label, sample_real_Clean,sample_real)
                from_real_film = True
                imgClean = resize_by_scale(imgClean, 1.8)
                imgDirty = resize_by_scale(imgDirty, 1.8)


                # cv2.imshow("imgClean",imgClean)
                # cv2.imshow("imgDirty",imgDirty)
                # cv2.waitKey(0)

            else:
                img = get_random(label, sample)
                from_real_film = False
                img = resize_by_scale(img, img_scale*0.8)
        
        point_1 = (0,0)
        point_2 = (0,0)
        if not from_real_film:
                # Augment the image
                img, rotation = augment(img, apply_rotation=True, apply_transformation=True, apply_boldness=True)
        else:
            #TODO Need to predefine rotations for real films.
            rotation = 0
            img, rotation, point_1, point_2=real_symbol_utils.add_real_symbol(imgClean,imgDirty,1)
    


        #? to fit the symbol to original symbol bounding box, but stretches symbol, so not correct. 
 
        symbolDim=(img.shape[1],img.shape[0])
        if(symbolDim[0]>dim[0] or symbolDim[1]>dim[1]):

            widthRatio=dim[0]/symbolDim[0]
            heightRatio=dim[1]/symbolDim[1]
            if(widthRatio<heightRatio):
                #todo use widthRation to resize
                img = cv2.resize(img, (int(round(img.shape[1]*widthRatio)),int(round(img.shape[0]*widthRatio))))
                symbolDim=(img.shape[1],img.shape[0])
                if(from_real_film):
                    point_1=(int(point_1[0]*widthRatio),int(point_1[1]*widthRatio))
                    point_2=(int(point_2[0]*widthRatio),int(point_2[1]*widthRatio))

                newPlacementX=dim[0]
                newPlacementY=dim[1]
            else:
                img = cv2.resize(img, (int(round(img.shape[1]*heightRatio)),int(round(img.shape[0]*heightRatio))))
                symbolDim=(img.shape[1],img.shape[0])
                if(from_real_film):
                    point_1=(int(point_1[0]*heightRatio),int(point_1[1]*heightRatio))
                    point_2=(int(point_2[0]*heightRatio),int(point_2[1]*heightRatio))
                newPlacementX=dim[0]
                newPlacementY=dim[1]
        

        



        labels.append(label)
        rotations.append(rotation)
        
        #Check if there is overlap with current symbols.
        #If there is overlap the generate new locations and check again.

        #? to counteract the -10 in get points and not to get a empty range
        newPlacementX=dim[0]+11
        newPlacementY=dim[1]+11
        point1, point2 = get_points((newPlacementY,newPlacementX), img, locations, locations_units,location_placement)
        point1=startRangePoint[1]+point1
        point2=startRangePoint[0]+point2


        #? checks if with plus 15 will it be out of canvas
        if(point1+img.shape[0]>canvas.shape[0]):
            point1-=15
        elif(point1+img.shape[0]<canvas.shape[0]):
            point1+=15
            
        if(point1+img.shape[1]>canvas.shape[1]):
            point2-=15
        elif(point1+img.shape[0]<canvas.shape[0]):
            point2+=15
            

        #We append upper left corner point and lower right corner point.
        locations.append(((point1+point_1[0],point2+point_1[1]),(point1+img.shape[0]+point_2[0],point2+img.shape[1]+point_2[1])))
        #If there is overlap we don't want to overwrite black pixels with white background.
        canvas = place_symbol(canvas, img, point1, point2)
 
    return canvas, locations, labels, rotations, locations_units, labels_units

def ProcessBackgrounds(backgroundPath):
    imagesPath=glob.glob(backgroundPath+"/*.jpg")
    imagesPath+=glob.glob(backgroundPath+"/*.png")
    backgroundImageList=[]
    for imagePath in imagesPath:
        imageName=os.path.basename(imagePath).split(".")[0]
        parentFolderPath=os.path.dirname(imagePath)
        image=cv2.imread(imagePath,0)
        removedLabelsImage=image.copy()
        imageshape=image.shape


        boundingBoxesToRemove=[]
        with open(parentFolderPath+"/"+imageName+".txt") as labelFile:
            for label in labelFile.readlines():
                label=label.split(" ")
                boundingBoxCoordinatesInfoNormalized=ParseBoundingBoxInfo(label)
                boundingBoxPoint=GetBoundingBoxPoints(boundingBoxCoordinatesInfoNormalized,imageshape)
                boundingBoxesToRemove.append(boundingBoxPoint)
        for boundingBox in boundingBoxesToRemove:
            startPoint=boundingBox[0]
            endPoint=boundingBox[1]
            #TODO dont forget to enable!!!!!!!!
            removedLabelsImage=cv2.rectangle(removedLabelsImage, startPoint, endPoint, (255, 255, 255), -1)

        dim=(imageshape[0],imageshape[1])
        backgroundImageList.append([removedLabelsImage,boundingBoxesToRemove,dim])
    return backgroundImageList

def ParseBoundingBoxInfo(label):
    boundingBoxCoordinates=[float(label[1]),float(label[2]),float(label[3]),float(label[4])]
    return boundingBoxCoordinates

def GetBoundingBoxPoints(boundingBoxCoordinatesInfoNormalized,imageshape):
    x,y,w,h=boundingBoxCoordinatesInfoNormalized

    imageW=imageshape[1]
    imageH=imageshape[0]

    x=int(x*imageW)
    w=int(w*imageW)

    y=int(y*imageH)
    h=int(h*imageH)

    symbolXStart=int(x-w/2)
    symbolyStart=int(y-h/2)
    symbolXEnd=int(x+w/2)
    symbolyEnd=int(y+h/2)
    return[(symbolXStart,symbolyStart),(symbolXEnd,symbolyEnd)]
