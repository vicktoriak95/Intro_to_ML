#################################
# Your name: Vicktoria Kraslavski
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    n = len(X_train)
    D = np.array([1 / n for i in range(n)])
    hypotheses = []
    alpha_vals = []

    for t in range(T):
        #print("t =", t)
        h_pred, h_index, h_theta, h_loss = full_weak_learner(X_train, y_train, D)
        #print("h is ok")
        e_t = h_loss
        w_t = 0.5 * np.log((1 - e_t) / e_t)
        prediction_vector = np.array([predict_label(x, h_pred, h_index, h_theta) for x in X_train])
        D = D_update(prediction_vector, y_train, D, w_t)
        hypotheses.append((h_pred, h_index, h_theta))
        alpha_vals.append(w_t)

    return hypotheses, alpha_vals


##############################################

def weak_learner(X_train, y_train, D, b):
    # Initializing
    best_loss = np.inf
    best_theta = -7
    best_index = -1
    d = len(X_train[0])
    n = len(X_train)
    data = list(zip(X_train, y_train, D))

    # Iterating over j options
    for index in range(d):
        data.sort(key=lambda x: x[0][index])  # Sorting by j's coordinate
        coor_vec = np.array([item[0][index] for item in data])
        coor_vec = np.concatenate((coor_vec, coor_vec[n-1]+1), axis=None)  # Adding x_nj+1 as last coordinate

        # For theta = x_1j-1
        loss = np.sum(np.array([item[2] for item in data if item[1] == b]))

        if loss < best_loss:
            best_loss = loss
            best_index = index
            best_theta = data[0][0][index] - 1  # x_1j - 1

        # Iterating over theta values
        for i in range(n):
            loss = loss - (b * data[i][1]*data[i][2])  # loss - y_i*D_i

            if (loss < best_loss) and (coor_vec[i] != coor_vec[i+1]):
                best_loss = loss
                best_index = index
                best_theta = 0.5*(coor_vec[i] + coor_vec[i+1])  # 0.5(x_i,j + x_i+1,j)

    return best_index, best_theta, best_loss

def full_weak_learner(X_train, y_train, D):
    index1, theta1, loss1 = weak_learner(X_train, y_train, D, 1)
    index2, theta2, loss2 = weak_learner(X_train, y_train, D, -1)

    if loss1 < loss2:
        return 1, index1, theta1, loss1
    else:
        return -1, index2, theta2, loss2


def D_update(prediction_vector, y_train, D, w_t):
    exp_vector = np.exp(-w_t * prediction_vector * y_train)
    Z = D @ exp_vector
    new_D = 1/Z * (D * exp_vector)
    return new_D


def predict_label(x, sign, j, theta):
    return sign * np.sign(theta - x[j])

def loss_calc(pred_vector, y):
    loss = 0
    for i in range(len(pred_vector)):
        if pred_vector[i] != y[i]:
            loss += 1
    return loss / len(pred_vector)

def predict_adaboost(X, hypotheses, alpha_vals):
    predict_vector = []
    for x in X:
        sum_vector = [alpha_vals[i] * predict_label(x, hypotheses[i][0], hypotheses[i][1], hypotheses[i][2]) for i in range(len(hypotheses))]
        tag = np.sum(sum_vector)
        if tag >= 0:
            label = 1
        else:
            label = -1
        predict_vector.append(label)
    return predict_vector

def calc_loss_wrt_t(X_train, y_train, X_test, y_test, hypotheses, alpha_vals):
    test_errors = []
    train_errors = []
    T = len(hypotheses)

    for t in range(T):
        #print("t = ", t)

        train_prediction = predict_adaboost(X_train, hypotheses[:t+1], alpha_vals[:t+1])
        test_prediction = predict_adaboost(X_test, hypotheses[:t+1], alpha_vals[:t+1])
        train_error = loss_calc(train_prediction, y_train)
        test_error = loss_calc(test_prediction, y_test)
        train_errors.append(train_error)
        test_errors.append(test_error)
    return train_errors, test_errors

def calc_exp_loss(X, y, hypotheses, alpha_vals):
    loss_vector = []
    m = len(X)
    exp_vector = np.zeros(m)
    T = len(hypotheses)
    for t in range(T):
        for i in range(m):
            exp_vector[i] += (-1) * y[i] * alpha_vals[t] * predict_label(X[i], hypotheses[t][0], hypotheses[t][1], hypotheses[t][2])
        loss = np.sum(np.exp(exp_vector))
        loss_vector.append((1 / m) * loss)

    return loss_vector


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    T = 80
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    #print("hypotheses: ", hypotheses)
    #print("alpha_vals: ", alpha_vals)
    """
    hypotheses = [(1, 26, 0.5), (-1, 31, 0.5), (-1, 22, 0.5), (1, 311, 0.5), (-1, 37, 1.5), (1, 372, 0.5),
                  (-1, 282, 0.5), (1, 292, 0.5), (-1, 196, 0.5), (1, 88, 0.5), (-1, 107, 0.5), (1, 402, 0.5),
                  (-1, 703, 0.5), (1, 804, 0.5), (-1, 557, 0.5), (1, 187, 0.5), (-1, 17, 1.5), (1, 189, 0.5),
                  (-1, 211, 0.5), (1, 311, 0.5), (-1, 195, 0.5), (1, 76, 0.5), (-1, 1, 2.5), (1, 23, 0.5),
                  (-1, 158, 0.5), (1, 232, 0.5), (-1, 179, 0.5), (1, 892, 0.5), (-1, 729, 0.5), (1, 741, 0.5),
                  (-1, 994, 0.5), (1, 822, 0.5), (-1, 501, 0.5), (1, 864, 0.5), (-1, 1026, 0.5), (1, 154, 0.5),
                  (-1, 15, 0.5), (1, 25, 0.5), (-1, 99, 0.5), (1, 246, 0.5), (-1, 301, 0.5), (1, 804, 0.5),
                  (-1, 371, 0.5), (1, 140, 0.5), (-1, 60, 0.5), (1, 26, 1.5), (-1, 1006, 0.5), (1, 275, 0.5),
                  (-1, 354, 0.5), (1, 379, 0.5), (-1, 156, 0.5), (1, 160, 0.5), (-1, 128, 0.5), (1, 4, 1.5),
                  (-1, 17, 0.5), (-1, 46, 0.5), (1, 41, 0.5), (-1, 29, 0.5), (1, 236, 0.5), (-1, 687, 0.5),
                  (1, 892, 0.5), (-1, 670, 0.5), (1, 383, 0.5), (-1, 709, 0.5), (1, 1012, 0.5), (-1, 498, 0.5),
                  (1, 435, 0.5), (-1, 994, 0.5), (1, 802, 0.5), (-1, 641, 0.5), (1, 55, 1.5), (-1, 198, 0.5),
                  (1, 62, 0.5), (-1, 24, 0.5), (1, 332, 0.5), (-1, 637, 0.5), (1, 1041, 0.5), (-1, 986, 0.5),
                  (1, 582, 0.5), (-1, 859, 0.5)]
    alpha_vals = [0.26151742796590843, 0.1841508459039694, 0.14910232134602863, 0.15469714068798077, 0.1915574866154928,
                  0.14314431361864305, 0.16148927520512862, 0.14977352552677523, 0.15264130425915018,
                  0.14298235499432413, 0.1232388554499457, 0.1238034143870661, 0.1406654300359237, 0.13407276133538495,
                  0.1336567712545831, 0.11670999682271353, 0.11128504234044315, 0.11584091244329471,
                  0.11725880867725799, 0.11837865544369736, 0.1154248188051812, 0.10386394418685344,
                  0.10230917669639172, 0.11359596807342957, 0.10318982662275046, 0.10787726802795057,
                  0.11005224530587043, 0.09951335574039732, 0.12661287072537064, 0.11290427803950069,
                  0.11694455759207543, 0.11802856069994117, 0.11287176966651957, 0.10606792805576688,
                  0.11099763404211005, 0.10733991471431656, 0.10129061101278855, 0.09846949896010765,
                  0.09429860880085342, 0.08902133857123443, 0.1079300841178513, 0.09717822686858121,
                  0.11056976082735676, 0.09792042517369205, 0.08953964951090948, 0.08644626308410479,
                  0.10280628868886399, 0.10459214117749982, 0.09225961785392627, 0.09814011977709979,
                  0.09613154551994836, 0.08323950913813724, 0.08950661923360151, 0.0857889981732911, 0.0919166146829436,
                  0.07666005900883781, 0.07848267550743773, 0.07524401762222457, 0.07473087773492901,
                  0.09944116883347706, 0.10016234752905902, 0.1028588948918129, 0.09165611106355137,
                  0.09690909125086616, 0.09333348914748903, 0.09523125399484374, 0.09067115806419819,
                  0.08532845242193139, 0.09445076110183356, 0.09358415170596975, 0.093523733622731, 0.08596147923770475,
                  0.08225938932509405, 0.07401478691950251, 0.06703934115345452, 0.0950814290369983,
                  0.08892531816431326, 0.09347871317344918, 0.09262498183750978, 0.08468446616954839]
    """
    train_errors, test_errors = calc_loss_wrt_t(X_train, y_train, X_test, y_test, hypotheses, alpha_vals)

    train_exploss = calc_exp_loss(X_train, y_train, hypotheses, alpha_vals)
    test_exploss = calc_exp_loss(X_test, y_test, hypotheses, alpha_vals)

    t_vector = [t for t in range(T)]

    plt.xlabel('t')
    plt.ylabel('Error')
    plt.plot(t_vector, train_errors, label="Train error", marker=".", color="blue")
    plt.plot(t_vector, test_errors, label="Test error", marker=".", color="purple")
    plt.legend()
    #plt.show()
    plt.xlabel('t')
    plt.ylabel('Avg exp-loss')
    plt.plot(t_vector, train_exploss, label="Train exp-loss", marker=".", color="blue")
    plt.plot(t_vector, test_exploss, label="Test exp-loss", marker=".", color="purple")
    plt.legend()
    #plt.show()



if __name__ == '__main__':
    main()



