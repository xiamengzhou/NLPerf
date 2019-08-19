def p2f(x):
    return float(x.strip('%'))/100


def convert_label(df):
    return df.values.reshape(len(df))
