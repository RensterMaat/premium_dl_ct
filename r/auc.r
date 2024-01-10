library(cvAUC)
library(ResourceSelection)
library(dplyr)

df = read.csv("/mnt/c/Users/user/data/results_dl_response/predictions.csv")
target = "response"

evaluate = function(predictor) {
    auc = ci.cvAUC(df[[predictor]], df[[target]], folds=df$center)
    auc$ci_lower = auc$ci[1]
    auc$ci_upper = auc$ci[2]
    auc = auc[-3]
    auc$hoslem = hoslem.test(df[[target]], df[[predictor]])$p.value

    return(auc)
}

output = data.frame(cbind(
    evaluate('calibrated_dl_preds'),
    evaluate('clinical_preds'),
    evaluate('combination_preds')
))


output =  output %>%
    rename(
        'calibrated_dl_preds' = 'X1',
        'clinical_preds' = 'X2',
        'combination_preds' = 'X3'
    )

output$calibrated_dl_preds = as.numeric(output$calibrated_dl_preds)
output$clinical_preds = as.numeric(output$clinical_preds)
output$combination_preds = as.numeric(output$combination_preds)


write.csv(
    output,
    "/home/rens/repos/premium_dl_ct/tables/auc.csv"
)
