import torch
import torch.utils.data
from tqdm import tqdm


def run_main_1(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion, lr_scheduler, fold):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}, LR {current_lr:.6f}", unit="batch")
        for ii, batch in enumerate(train_bar):
            mri_images = batch.get("mri")
            pet_images = batch.get("pet")
            cli_tab = batch.get("clinical")
            label = batch.get("label")
            if torch.isnan(mri_images).any():
                print("train: NaN detected in input mri_images")
            if torch.isnan(pet_images).any():
                print("train: NaN detected in input pet_images")
            mri_images = mri_images.to(device)
            pet_images = pet_images.to(device)
            cli_tab = cli_tab.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            mri_feature, pet_feature, cli_feature, outputs_logit = model.forward(mri_images, pet_images, cli_tab)
            loss = criterion(mri_feature, pet_feature, cli_feature, label, outputs_logit)
            prob = torch.softmax(outputs_logit, dim=1)
            _, predictions = torch.max(prob, dim=1)
            observer.train_update(loss, prob, predictions, label)
            loss.backward()
            optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, desc=f"Evaluating Epoch {epoch + 1}", unit="batch")
            for i, batch in enumerate(test_bar):
                mri_images = batch.get("mri").to(device)
                pet_images = batch.get("pet").to(device)
                cli_tab = batch.get("clinical").to(device)
                label = batch.get("label").to(device)
                mri_feature, pet_feature, cli_feature, outputs_logit = model.forward(mri_images, pet_images, cli_tab)
                loss = criterion(mri_feature, pet_feature, cli_feature, label, outputs_logit)
                prob = torch.softmax(outputs_logit, dim=1)
                _, predictions = torch.max(prob, dim=1)
                observer.eval_update(loss, prob, predictions, label)
        if observer.execute(epoch + 1, epochs, len(train_loader.dataset), len(test_loader.dataset), fold, model=model):
            print("Early stopping")
            break
    observer.finish(fold)


def run_main_for_hfbsurve(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion,
                          lr_scheduler, fold):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}, LR {current_lr:.6f}", unit="batch")
        for ii, batch in enumerate(train_bar):
            mri_images = batch.get("mri").to(device)
            pet_images = batch.get("pet").to(device)
            cli_tab = batch.get("clinical").to(device)
            label = batch.get("label").to(device)
            optimizer.zero_grad()
            outputs_logit = model(mri_images, pet_images, cli_tab)
            loss = criterion(outputs_logit, label)
            prob = torch.softmax(outputs_logit, dim=1)
            _, predictions = torch.max(prob, dim=1)
            loss.backward()
            optimizer.step()
            observer.train_update(loss, prob, predictions, label)
        if lr_scheduler:
            lr_scheduler.step()
        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, desc=f"Evaluating Epoch {epoch + 1}", unit="batch")
            for i, batch in enumerate(test_bar):
                mri_images = batch.get("mri").to(device)
                pet_images = batch.get("pet").to(device)
                cli_tab = batch.get("clinical").to(device)
                label = batch.get("label").to(device)
                outputs_logit = model(mri_images, pet_images, cli_tab)
                loss = criterion(outputs_logit, label)
                prob = torch.softmax(outputs_logit, dim=1)
                _, predictions = torch.max(prob, dim=1)
                observer.eval_update(loss, prob, predictions, label)
        if observer.execute(epoch + 1, epochs, len(train_loader.dataset), len(test_loader.dataset), fold, model=model):
            print("Early stopping")
            break
    observer.finish(fold)


def run_main_for_IMF(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion, lr_scheduler,
                     fold):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}, LR {current_lr:.6f}", unit="batch")
        for ii, batch in enumerate(train_bar):
            mri_images = batch.get("mri").to(device)
            pet_images = batch.get("pet").to(device)
            cli_tab = batch.get("clinical").to(device)
            label = batch.get("label").to(device)
            label_2d = batch.get("label_2d").to(device)
            if torch.isnan(mri_images).any():
                print("train: NaN detected in input mri_images")
            if torch.isnan(pet_images).any():
                print("train: NaN detected in input pet_images")
            optimizer.zero_grad()
            outputs = model.forward(mri_images, pet_images, cli_tab)
            loss = criterion(outputs, label_2d)
            prob = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4.0
            _, predictions = torch.max(prob, dim=1)
            loss.backward()
            optimizer.step()
            observer.train_update(loss, prob, predictions, label)
        if lr_scheduler:
            lr_scheduler.step()
        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, desc=f"Evaluating Epoch {epoch + 1}", unit="batch")
            for i, batch in enumerate(test_bar):
                mri_images = batch.get("mri").to(device)
                pet_images = batch.get("pet").to(device)
                cli_tab = batch.get("clinical").to(device)
                label = batch.get("label").to(device)
                label_2d = batch.get("label_2d").to(device)
                outputs = model.forward(mri_images, pet_images, cli_tab)
                loss = criterion(outputs, label_2d)
                prob = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4.0
                _, predictions = torch.max(prob, dim=1)
                observer.eval_update(loss, prob, predictions, label)
        if observer.execute(epoch + 1, epochs, len(train_loader.dataset), len(test_loader.dataset), fold, model=model):
            print("Early stopping")
            break
    observer.finish(fold)


def run_main_for_MDL(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion, lr_scheduler,
                     fold):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        current_lr = optimizer.param_groups[0]['lr']

        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}, LR {current_lr:.6f}", unit="batch")

        for ii, batch in enumerate(train_bar):
            gm_img_torch = batch.get("gm")
            wm_img_torch = batch.get("wm")
            pet_img_torch = batch.get("pet")
            label = batch.get("label")
            if torch.isnan(gm_img_torch).any():
                print("train: NaN detected in input mri_images")
            if torch.isnan(wm_img_torch).any():
                print("train: NaN detected in input pet_images")
            if torch.isnan(pet_img_torch).any():
                print("train: NaN detected in input pet_images")
            gm_img_torch = gm_img_torch.to(device)
            wm_img_torch = wm_img_torch.to(device)
            pet_img_torch = pet_img_torch.to(device)
            input_data = torch.concat([gm_img_torch, wm_img_torch, pet_img_torch], dim=1)
            label = label.to(device)
            optimizer.zero_grad()
            outputs, roi_out = model(input_data)
            loss = criterion(outputs, label)
            prob = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(prob, dim=1)
            observer.train_update(loss, prob, predictions, label)
            loss.backward()
            optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, desc=f"Evaluating Epoch {epoch + 1}", unit="batch")
            for i, batch in enumerate(test_bar):
                gm_img_torch = batch.get("gm")
                wm_img_torch = batch.get("wm")
                pet_img_torch = batch.get("pet")
                label = batch.get("label")
                gm_img_torch = gm_img_torch.to(device)
                wm_img_torch = wm_img_torch.to(device)
                pet_img_torch = pet_img_torch.to(device)
                input_data = torch.concat([gm_img_torch, wm_img_torch, pet_img_torch], dim=1)
                label = label.to(device)
                outputs, roi_out = model(input_data)
                loss = criterion(outputs, label)
                prob = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(prob, dim=1)
                observer.eval_update(loss, prob, predictions, label)
        if observer.execute(epoch + 1, epochs, len(train_loader.dataset), len(test_loader.dataset), fold, model=model):
            print("Early stopping")
            break
    observer.finish(fold)


def run_main_for_RLAD(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion, lr_scheduler,
                      fold):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}, LR {current_lr:.6f}", unit="batch")
        for ii, batch in enumerate(train_bar):
            mri_images = batch.get("mri").to(device)
            label = batch.get("label").to(device)
            if torch.isnan(mri_images).any():
                print("train: NaN detected in input mri_images")
            optimizer.zero_grad()
            _, outputs, _ = model(mri_images)
            loss = criterion(outputs, label)
            prob = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(prob, dim=1)
            observer.train_update(loss, prob, predictions, label)
            loss.backward()
            optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, desc=f"Evaluating Epoch {epoch + 1}", unit="batch")
            for i, batch in enumerate(test_bar):
                mri_images = batch.get("mri").to(device)
                label = batch.get("label").to(device)
                _, outputs, _ = model(mri_images)
                loss = criterion(outputs, label)
                prob = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(prob, dim=1)
                observer.eval_update(loss, prob, predictions, label)
        if observer.execute(epoch + 1, epochs, len(train_loader.dataset), len(test_loader.dataset), fold, model=model):
            print("Early stopping")
            break
    observer.finish(fold)


def run_main_for_resnet(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion, lr_scheduler,
                        fold):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}, LR {current_lr:.6f}", unit="batch")
        for ii, batch in enumerate(train_bar):
            mri_images = batch.get("mri").to(device)
            pet_images = batch.get("pet").to(device)
            label = batch.get("label").to(device)
            if torch.isnan(mri_images).any():
                print("train: NaN detected in input mri_images")
            if torch.isnan(pet_images).any():
                print("train: NaN detected in input pet_images")
            optimizer.zero_grad()
            mri_pet_images = torch.concat([mri_images, pet_images], dim=1)
            outputs_logit = model(mri_pet_images)
            loss = criterion(outputs_logit, label)
            prob = torch.softmax(outputs_logit, dim=1)
            _, predictions = torch.max(prob, dim=1)
            loss.backward()
            optimizer.step()
            observer.train_update(loss, prob, predictions, label)
        if lr_scheduler:
            lr_scheduler.step()
        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, desc=f"Evaluating Epoch {epoch + 1}", unit="batch")
            for i, batch in enumerate(test_bar):
                mri_images = batch.get("mri").to(device)
                pet_images = batch.get("pet").to(device)
                label = batch.get("label").to(device)
                mri_pet_images = torch.concat([mri_images, pet_images], dim=1)
                outputs_logit = model(mri_pet_images)
                loss = criterion(outputs_logit, label)
                prob = torch.softmax(outputs_logit, dim=1)
                _, predictions = torch.max(prob, dim=1)
                observer.eval_update(loss, prob, predictions, label)
        if observer.execute(epoch + 1, epochs, len(train_loader.dataset), len(test_loader.dataset), fold, model=model):
            print("Early stopping")
            break
    observer.finish(fold)


def run_main_for_trilight_net(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion,
                             lr_scheduler, fold):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}, LR {current_lr:.6f}", unit="batch")
        for ii, batch in enumerate(train_bar):
            mri_images = batch.get("mri").to(device)
            pet_images = batch.get("pet").to(device)
            cli_tab = batch.get("clinical").to(device)
            label = batch.get("label").to(device)
            optimizer.zero_grad()
            mri_images = mri_images.to(device)
            pet_images = pet_images.to(device)
            cli_tab = cli_tab.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            """
            {
            "output": output,  # 保持[B, num_classes]形状
            "gate_scores": gate_scores,
            "expert_indices": top_k_indices
            }
            """
            # moe_output = model(mri_images, pet_images, cli_tab)
            # outputs_logit = moe_output["output"]

            outputs_logit = model(mri_images, pet_images, cli_tab)
            prob = torch.softmax(outputs_logit, dim=1)
            cls_loss = criterion(prob, label)
            # aux_loss = model.classify_head.compute_aux_loss(moe_output["gate_scores"], moe_output["expert_indices"])
            # loss = cls_loss + aux_loss
            loss = cls_loss
            _, predictions = torch.max(prob, dim=1)
            loss.backward()
            optimizer.step()
            observer.train_update(loss, prob, predictions, label)
        if lr_scheduler:
            lr_scheduler.step()
        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, desc=f"Evaluating Epoch {epoch + 1}", unit="batch")
            for i, batch in enumerate(test_bar):
                mri_images = batch.get("mri").to(device)
                pet_images = batch.get("pet").to(device)
                cli_tab = batch.get("clinical").to(device)
                label = batch.get("label").to(device)
                mri_images = mri_images.to(device)
                pet_images = pet_images.to(device)
                cli_tab = cli_tab.to(device)
                label = label.to(device)
                # moe_output = model(mri_images, pet_images, cli_tab)
                # outputs_logit = moe_output["output"]
                outputs_logit = model(mri_images, pet_images, cli_tab)
                prob = torch.softmax(outputs_logit, dim=1)
                cls_loss = criterion(prob, label)

                # aux_loss = model.classify_head.compute_aux_loss(moe_output["gate_scores"], moe_output["expert_indices"])
                # loss = cls_loss + aux_loss
                loss = cls_loss
                _, predictions = torch.max(prob, dim=1)
                observer.eval_update(loss, prob, predictions, label)
        if observer.execute(epoch + 1, epochs, len(train_loader.dataset), len(test_loader.dataset), fold, model=model):
            print("Early stopping")
            break
    observer.finish(fold)
