from nanograd.nn import MLP

if __name__ == "__main__":
    model = MLP(3, [4,4,1])

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]

    ys = [1.0, -1.0, -1.0, 1.0]

    for k in range(500):
        # forward pass
        y_pred = [model(x) for x in xs]
        loss = sum((y_out - y_gt)**2 for y_gt, y_out in zip(ys, y_pred))

        # backward pass
        model.zero_grad()
        loss.backward()
        
        # update
        for p in model.parameters():
            p.data += -0.03 * p.grad
        
        if k % 100 == 0:
            print(k, loss.data)

    print(y_pred)