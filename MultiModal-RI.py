import sys
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import preprocess as pp
import pickle
from kan_RBF import MultiModalRBFKAAGNN, MultiModalRBFKAAGNNWithPretrainedBERT


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5, verbose=True
        )

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0

        accumulation_steps = 4
        self.optimizer.zero_grad()

        for i, idx in enumerate(range(0, N, batch_train)):
            data_batch = list(zip(*dataset[idx:idx + batch_train]))
            loss = self.model.forward_regressor(data_batch, train=True)

            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            loss_total += loss.item() * accumulation_steps

        return loss_total / len(dataset)

class Tester(object):
    def __init__(self, model):
        self.model = model

    def test_regressor(self, dataset):
        N = len(dataset)
        SMILES, Ts, Ys = '', [], []
        SAE = 0 
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i + batch_test]))
            (Smiles, predicted_values, correct_values) = self.model.forward_regressor(
                data_batch, train=False)
            SMILES += ' '.join(Smiles) + ' '
            Ts.append(correct_values)
            Ys.append(predicted_values)

            SAE += sum(np.abs(predicted_values - correct_values))
        SMILES = SMILES.strip().split()
        T, Y = map(str, np.concatenate(Ts)), map(str, np.concatenate(Ys))
        predictions = '\n'.join(['\t'.join(x) for x in zip(SMILES, T, Y)])
        MAEs = SAE / N 
        return MAEs, predictions

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write(MAEs + '\n')

    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:
            f.write('Smiles\tCorrect\tPredict\n')
            f.write(predictions + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def split_dataset(dataset, ratio):
    np.random.seed(1234) 
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)


if __name__ == "__main__":
    radius = 1
    dim = 96
    layer_hidden = 4
    layer_output = 4
    batch_train = 32
    batch_test = 32
    lr = 1e-3
    lr_decay = 0.85
    decay_interval = 40
    iteration = 400 
    N = 10000
    
    bert_dim = 512
    bert_heads = 16
    bert_layers = 8
    fusion_type = 'cross_attention' # 'concat', 'cross_attention', 'attention', 'gated', 'bilinear'
    fusion_heads = 6
    use_pretrained_bert = False
    pretrained_model_name = 'seyonec/ChemBERTa-zinc-base-v1'
    
    use_augmentation = True
    num_augments = 4
    augmentation_types = ['random', 'canonical', 'kekulize', 'isomeric']
    
    path = './data/'
    dataname = 'in-house'
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    
    if use_augmentation:
        print(f'Using SMILES augmentation with {num_augments} augmentations per molecule')
        dataset_train = pp.create_dataset_with_augmentation(
            'train_set_stratified.txt', path, dataname, 
            num_augments=num_augments, 
            augmentation_types=augmentation_types
        )
    else:
        dataset_train = pp.create_dataset('train_set_stratified.txt', path, dataname)
    dataset_train, dataset_dev = split_dataset(dataset_train, 0.9)
    dataset_test = pp.create_dataset('test_set_stratified.txt', path, dataname)

    lr, lr_decay = map(float, [lr, lr_decay])
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    print('-' * 100)
    print('Just a moment......')
    print('-' * 100)
    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    print('# of development data samples:', len(dataset_dev))
    print('# of test data samples:', len(dataset_test))
    print('-' * 100)
    print('Creating a multi-modal model.')
    
    torch.manual_seed(123)
    
    vocab_size = len(pp.smiles_tokenizer) if pp.smiles_tokenizer else 1000
    
    if use_pretrained_bert:
        print(f'Using pretrained BERT model: {pretrained_model_name}')
        model = MultiModalRBFKAAGNNWithPretrainedBERT(
            N, dim, layer_hidden, layer_output,
            pretrained_model_name=pretrained_model_name,
            num_centers=6,
            rbf_type='inverse_quadratic',
            heads=4,
            fusion_type=fusion_type,
            fusion_heads=fusion_heads,
            freeze_bert=True
        ).to(device)
    else:
        print('Using custom MolBERT model')
        model = MultiModalRBFKAAGNN(
            N, dim, layer_hidden, layer_output, vocab_size,
            bert_dim=bert_dim,
            bert_heads=bert_heads,
            bert_layers=bert_layers,
            num_centers=6,
            rbf_type='inverse_quadratic',
            heads=4,
            fusion_type=fusion_type,
            fusion_heads=fusion_heads
        ).to(device)
    
    trainer = Trainer(model)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-' * 100)
    
    file_MAEs = path + 'MAEs_multimodal' + '.txt'
    file_test_result = path + 'test_prediction_multimodal' + '.txt'
    file_predictions = path + 'train_prediction_multimodal' + '.txt'
    file_model = path + 'inhouse_model_multimodal' + '.h5'
    result = 'Epoch\tTime(sec)\tLoss_train\tMAE_train\tMAE_dev\tMAE_test'
    with open(file_MAEs, 'w') as f:
        f.write(result + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')
    np.random.seed(1234)
    start = timeit.default_timer()

    MAE_best = 9999999
    
    for epoch in range(iteration):

        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay
        model.train()
        loss_train = trainer.train(dataset_train)
        model.eval()
        MAE_train, predictions_train = tester.test_regressor(dataset_train)
        MAE_dev = tester.test_regressor(dataset_dev)[0]
        MAE_test = tester.test_regressor(dataset_test)[0]

        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-' * 100)
            print(result)

        results = '\t'.join(map(str, [epoch, time, loss_train, MAE_train,
                                      MAE_dev, MAE_test]))
        tester.save_MAEs(results, file_MAEs)
        if MAE_dev <= MAE_best:
            MAE_best = MAE_dev
            tester.save_model(model, file_model)
        print(results)
    
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import median_absolute_error, r2_score, mean_absolute_error, mean_squared_error


    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))


    loss = pd.read_table(file_MAEs)
    plt.plot(loss['MAE_train'], color='r', label='MAE of train set')
    plt.plot(loss['MAE_dev'], color='b', label='MAE of validation set')
    plt.plot(loss['MAE_test'], color='y', label='MAE of test set')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(path + 'loss_multimodal.tif', dpi=300)
    plt.show()
    
    predictions_train = tester.test_regressor(dataset_train)[1]
    tester.save_predictions(predictions_train, file_predictions)
    predictions_test = tester.test_regressor(dataset_test)[1]
    tester.save_predictions(predictions_test, file_test_result)
    
    res = pd.read_table(file_test_result)
    r2 = r2_score(res['Correct'], res['Predict'])
    mae = mean_absolute_error(res['Correct'], res['Predict'])
    mse = mean_squared_error(res['Correct'], res['Predict'])
    medae = median_absolute_error(res['Correct'], res['Predict'])
    rmae = np.mean(np.abs(res['Correct'] - res['Predict']) / res['Correct']) * 100
    median_re = np.median(np.abs(res['Correct'] - res['Predict']) / res['Correct'])
    mean_re = np.mean(np.abs(res['Correct'] - res['Predict']) / res['Correct'])
    print(mae, mse, r2, medae)
    plt.plot(res['Correct'], res['Predict'], '.', color='blue')
    plt.plot([0, 2000], [0, 2000], color='red')
    plt.ylabel('Predicted RI')
    plt.xlabel('Experimental RI')
    plt.text(0, 2000, 'R2=' + str(round(r2, 4)), fontsize=12)
    plt.text(800, 2000, 'MAE=' + str(round(mae, 4)), fontsize=12)
    plt.text(0, 1800, 'MedAE=' + str(round(medae, 4)), fontsize=12)
    plt.text(800, 1800, 'MRE=' + str(round(mean_re, 4)), fontsize=12)
    plt.text(0, 1600, 'MedRE=' + str(round(median_re, 4)), fontsize=12)
    plt.savefig(path + 'c-p_multimodal.tif', dpi=300)
    plt.show()


    print('-' * 100)
    print('Loading the best model...')
    model.load_state_dict(torch.load(file_model))
    model.eval()

    print('Re-evaluating with the best model...')
    tester = Tester(model)
    predictions_test = tester.test_regressor(dataset_test)[1]

    best_file_test_result = path + 'best_model_test_prediction_multimodal' + '.txt'
    tester.save_predictions(predictions_test, best_file_test_result)

    predictions_train = tester.test_regressor(dataset_train)[1]
    best_file_predictions = path + 'best_model_train_prediction_multimodal' + '.txt'
    tester.save_predictions(predictions_train, best_file_predictions)

    res = pd.read_table(best_file_test_result)
    r2 = r2_score(res['Correct'], res['Predict'])
    mae = mean_absolute_error(res['Correct'], res['Predict'])
    mse = mean_squared_error(res['Correct'], res['Predict'])
    medae = median_absolute_error(res['Correct'], res['Predict'])
    rmae = np.mean(np.abs(res['Correct'] - res['Predict']) / res['Correct']) * 100
    median_re = np.median(np.abs(res['Correct'] - res['Predict']) / res['Correct'])
    mean_re = np.mean(np.abs(res['Correct'] - res['Predict']) / res['Correct'])

    print('=' * 100)
    print('BEST MODEL PERFORMANCE (Multi-Modal):')
    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'MedAE: {medae:.4f}')
    print(f'Mean Relative Error: {mean_re:.4f}')
    print(f'Median Relative Error: {median_re:.4f}')
    print('=' * 100)
