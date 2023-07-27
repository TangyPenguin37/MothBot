# Findings (so far)

Attemped removing some columns to see if they were useful or not. Also tried grouping the wings together to see if that would help.

The numbers show the average accuracy of the model over 1000 runs (or where there are 2 numbers, I ran the model another 1000 times for no reason other than to see if the results were consistent).

## Only using one set of common measurements (Grouping: 1)

### LDA

- **Using ventral measurements: 0.8974285714285714**
- Using dorsal measurements: 0.8955584415584416
- Using both measurements: 0.8955324675324674

### LDA Multiclass

- **Using ventral measurements: 0.7762946428571429**
- Using dorsal measurements: 0.7751785714285714
- Using both measurements: 0.7757589285714288

**slightly better with ventral?**

## Ignoring colour percentages

### LDA

- Dorsal percentages: 0.8964649350649351/0.8963246753246752
- Ventral percentages: 0.8960129870129869/0.8965480519480519
- **No percentages: 0.8965766233766232/0.8971116883116882**
- Both percentages: 0.8965532467532467/0.8963558441558441

### LDA Multiclass

- Dorsal percentages: 0.7751183035714286/0.7773415178571429
- Ventral percentages: 0.7775580357142858/0.7789017857142857
- No percentages: 0.7768370535714285/0.7768928571428572
- **Both percentages: 0.7784196428571429/0.7786897321428571**

**Biggest changes**

## Reducing EFD orders

### LDA

- **no EFD: 0.8992961038961039**
- 1st EFD: 0.8987974025974027
- all EFDs: 0.8958051948051948

### LDA Multiclass

- **no EFD: 0.7867209821428571/0.7847120535714286**
- 1st EFD: 0.7847008928571428/0.7838861607142856
- all EFDs: 0.7791227678571429/0.779046875

**No EFD works best**

## Ignoring Aspect Ratio

### LDA

- **No AR: 0.8966935064935064/0.896864935064935**
- AR: 0.895264935064935/0.8959792207792207

### LDA Multiclass

- **No AR: 0.7786584821428572/0.7785022321428571/0.7784866071428571**
- AR: 0.7774620535714285/0.7775669642857143/0.7774732142857143

**No AR works best**

## Ignoring Circularity

### LDA

- No Circ: 0.8929714285714285/0.8935142857142857
- **Circ: 0.8962051948051948/0.8954233766233767**

### LDA Multiclass

- No Circ: 0.7776785714285713/0.7770066964285715
- **Circ: 0.7781919642857142/0.7778370535714286**

**Circ works best**

## Ignoring Location and Side

### LDA

- Neither: 0.8921194805194804/0.8921480519480519
- Just Loc: 0.8947324675324675/0.8950415584415584
- Just Side: 0.8953766233766234/0.8952701298701298
- **Both: 0.8955142857142857/0.8959454545454545**

### LDA Multiclass

- Neither: 0.7738816964285713/0.7732700892857143
- Just Loc: 0.7772834821428571/0.7778013392857143
- Just Side: 0.7767879464285714/0.7771227678571428
- Both: 0.776765625/0.7768504464285714

# Grouping Wings

## LDA

### By Wing

- **Left Front: 0.8802040816326532/0.8820918367346938**
- Right Front: 0.8599897959183673/0.8575
- Left Rear: 0.8755473684210526/0.8753368421052632
- Right Rear: 0.8774375/0.8777708333333333

### By Side

- Left: 0.8850466321243521/0.8841347150259065
- **Right: 0.8864974093264246/0.8881968911917095**

### By Location

- Front: 0.8850974358974362/0.884466666666667
- **Rear: 0.8968/0.8968631578947369**

### **Altogether - 0.8956649350649349/0.8962363636363636**

## LDA Multiclass

### By Wing

- Left Front: 0.7416929824561402/0.7414298245614035
- Right Front: 0.7218318584070796/0.725787610619469
- Left Rear: 0.7274684684684685/0.7317567567567569
- **Right Rear: 0.752801801801802/0.7523153153153153**

### By Side

- Left: 0.74978125/0.7508616071428571
- **Right: 0.7616026785714286/0.7635625**

### By Location

- Front: 0.7610176211453745/0.7608281938325991
- **Rear: 0.7844099099099099/0.7831261261261261**

### Altogether

- **0.7774397321428571/0.7768616071428571**
