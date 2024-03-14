import matplotlib.pyplot as plt

def simple_backtest(curr_values, pred_classes, use_step=10):

    def get_change(current, previous):
        if current == previous:
            return 1.0
        try:
            return (current - previous) / previous
        except ZeroDivisionError:
            return 0.

    capital = 1000
    transaction_cost = 0.01
    state = "short"
    action_count = 0
    prev_action = -1
    cap_over_time = [capital]
    for seq_idx in range(curr_values.shape[0]):

        action = pred_classes[seq_idx, use_step]
        if action == prev_action:
            action_count += 1

        if action == 1 and state == "short":
            if action_count > 1:
                buy_price = curr_values[seq_idx]
                state = "long"
                pct_change = 1 - transaction_cost
                action_count = 0
        elif action == 0 and state == "long":
            if action_count > 1:
                sell_price = curr_values[seq_idx]
                state = "short"
                pct_change = 1 + get_change(sell_price, buy_price) - transaction_cost
                action_count = 0

        elif action == 1 and state == "long":
            pct_change = 1.0
        elif action == 0 and state == "short":
            pct_change = 1.0

        curr_cap = cap_over_time[seq_idx] * pct_change
        cap_over_time.append(curr_cap)
        prev_action = action

    plt.figure()
    plt.plot(cap_over_time[0:5000])
    plt.show()

