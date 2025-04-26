import torch
import torch.nn.functional as F


def pad_and_stack(img_list):
    """
    Args
    ----
    img_list : list[torch.Tensor]
        Each tensor is shape (C, H, W), dtype/ device can differ.

    Returns
    -------
    batch : torch.Tensor
        Shape (B, C, H_max, W_max) on the same device / dtype as the first image.
    """

    print("*************")
    for img in img_list:
        print(img.shape)

    # --- 1. work out the target size ----------------------------------------
    h_max = max(img.shape[1] for img in img_list)
    w_max = max(img.shape[2] for img in img_list)

    if h_max > w_max:
        w_max = h_max
    elif w_max > h_max:
        h_max = w_max

    # --- 2. pad every image to (h_max, w_max) --------------------------------
    padded_imgs = []
    for img in img_list:
        # pad width dimension (left_pad, right_pad) then height (top_pad, bottom_pad)
        pad_w = w_max - img.shape[2]
        pad_h = h_max - img.shape[1]
        # we pad only on the *right* and *bottom*; change the tuple if you want symmetry
        padded = F.pad(img, (0, pad_w,     # width  : (left, right)
                             0, pad_h))    # height : (top,  bottom)
        padded_imgs.append(padded)

    # --- 3. stack into batch --------------------------------------------------
    batch = torch.stack(padded_imgs, dim=0)

    print(batch.shape)

    return batch