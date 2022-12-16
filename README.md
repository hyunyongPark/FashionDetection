# Kfashion Detection - YoloV5

### Reference Github
This is a github created by referring to the **yolov5 open source code of the ultraytics**.
Some modifications have been made to fit the Kfashion dataset.
- https://github.com/ultralytics/yolov5


### Model Description 
<table>
    <thead>
        <tr>
            <td>Model Architecture</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/FashionDetection/blob/main/img/architecture.png"/></td>
        </tr>
    </tbody>
</table>



### Requirements
- python V  # python version : 3.8.13
- dgl==0.9.1
- tqdm
- torch==1.9.1
- torchvision==0.10.1
- torchaudio==0.9.1
- torchtext==0.10.1
- dask
- partd
- pandas
- fsspec==0.3.3
- scipy
- sklearn



### cmd running

The install cmd is:
```
conda create -n your_prjname python=3.8
conda activate your_prjname
cd {Repo Directory}
pip install -r requirements.txt
```
- your_prjname : Name of the virtual environment to create


If you want to proceed with the new training, adjust the parameters and set the directory and proceed with the command below.

The Training cmd is:
```

python3 train.py 

```

The testing cmd is: 
```

python3 test.py 

```

The inferance cmd is: 
```

python3 detect.py 

```


<table>
    <thead>
        <tr>
            <td>Training example</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/FashionDetection/blob/main/img/training_ex.png"/></td>
        </tr>
    </tbody>
</table>


### Test Result

###### Testset Distribution
<table>
    <thead>
        <tr>
            <td>testset fashion category</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/FashionDetection/blob/main/img/distribution.png"/></td>
        </tr>
    </tbody>
</table>


- Model Performance Table

###### Bounding box test performance
|Model|Class Num|Testset Num|mAP@0.5|mAP@0.5:0.95|
|---|---|---|---|---|
|Cascade mask rcnn|21|250|*81.48%*|-|
|YoloV5|21|250|**94.1%**|**83.9%**|

###### Category classification test performance
Although the segmentation model had a slightly higher recall score for classification, it took 3 seconds to process detection per page in terms of service.
For object detection in yolov5, it takes less than 1 second per page.
|Model|Testset Num|Top3 Recall|
|---|---|---|
|Cascade mask rcnn|54,762|*93.4%*|
|YoloV5|54,760|**91.1%**|


|Class|Number|Top3 Recall|
|---|---|---|
|cardigan|1,450|*81.3%*|
|knitwear|3,527|*77.4%*|
|dress|9,649|*97.1%*|
|leggings|248|*85.1%*|
|vest|833|*72.7%*|
|bratop|80|*51.3%*|
|blouse|4,826|*89.6%*|
|shirt|1,922|*84.1%*|
|skirt|4,292|*90.7%*|
|jacket|1,783|*89.8%*|
|jumper|721|*73.8%*|
|jumpsuit|332|*93.4%*|
|jogger pants|198|*71.2%*|
|zipup|234|*63.7%*|
|jean|4,360|*84.5%*|
|coat|1,205|*71.2%*|
|tops|2,309|*59.4%*|
|t-shirt|7,837|*88.6%*|
|padded jacket|423|*64.3%*|
|pants|7,637|*86.4%*|
|hoody|675|*90.5%*|


- Example 
<table>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/FashionDetection/blob/main/img/ex1.png"/></td>
            <td><img src="https://github.com/hyunyongPark/FashionDetection/blob/main/img/ex2.png"/></td>
            <td><img src="https://github.com/hyunyongPark/FashionDetection/blob/main/img/ex3.png"/></td>
            <td><img src="https://github.com/hyunyongPark/FashionDetection/blob/main/img/ex4.png"/></td>
        </tr>
    </tbody>
</table>
