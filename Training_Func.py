
def  training_cycle(RED, training_time, optimizer, criterion, dataset ):
    
    graphic1 = []
    graphic2 = []
    running_epoch = []

    for epoch in range(training_time):
        train_loss = 0.0
        valid_loss = 0.0

        for inputs, labels in dataset["train_dataloader"]:
            optimizer.zero_grad()
            outputs = RED(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        RED.eval()
        for inputs, labels in dataset["validation_dataloader"]:
            outputs = RED(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

        train_loss = train_loss / len(dataset["train_dataloader"])
        print("[Para la epoca", epoch + 1, "] loss:", train_loss)
        graphic1.append(train_loss)

        valid_loss = valid_loss / len(dataset["validation_dataloader"])
        print("[Para la epoca", epoch + 1, "] val_loss:", valid_loss)
        graphic2.append(valid_loss)

        running_epoch.append(float(epoch))
    return graphic1, graphic2, running_epoch