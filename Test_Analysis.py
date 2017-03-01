from sklearn import metrics
#accuracy analysis
def test_analysis(predicted,y):
    # 计算整体准确率
    accuracy = metrics.accuracy_score(y, predicted)
    print('accuracy: %.2f%%' % (100 * accuracy))
    # 计算“0”样本的准确率
    count_y = 0
    count_P = 0
    for j in range(1, 298):
        if y[j] == 0:
            count_y += 1
            if predicted[j] == 0:
                count_P += 1
    accuracy_really = count_P / count_y
    print('accuracy_really: %.2f%%' % (100 * accuracy_really))