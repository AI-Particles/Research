import numpy as np
import torch
import torch.nn as nn



torch_float = torch.float32


def coord_down_(x, N=6):
    ratio = (N - 1.0) / N
    return (x - 0.3) * ratio
    #return ((p / 2.0 - np.array([1.25,0.75])) + np.array([0.5,0.5])).astype(np.int)


def coord_up_(x, N=6):
    ratio = N / (N - 1.0)
    return x * ratio + 0.3
    #return ((p + np.array([1.25,0.75])) * 2.0 + np.array([0.5,0.5])).astype(np.int)


def rect_up(rect_l, rect_t, rect_r, rect_b, levels=None, N=6):
    if levels is None:
        levels = 1
    if type(levels) is not int:
        levels_cnt = levels.max()
    else:
        levels_cnt = levels
    for i in range(levels_cnt):
        need_scaling = (levels > 0)
        rect_l = torch.where(need_scaling, coord_up_(rect_l, N), rect_l)
        rect_t = torch.where(need_scaling, coord_up_(rect_t, N), rect_t)
        rect_r = torch.where(need_scaling, coord_up_(rect_r, N), rect_r)
        rect_b = torch.where(need_scaling, coord_up_(rect_b, N), rect_b)
        levels -= 1
    return rect_l, rect_t, rect_r, rect_b


def rect_down(rect_l, rect_t, rect_r, rect_b, levels=None, N=6):
    if levels is None:
        levels = 1
    for i in range(levels):
        rect_l = coord_down_(rect_l, N)
        rect_t = coord_down_(rect_t, N)
        rect_r = coord_down_(rect_r, N)
        rect_b = coord_down_(rect_b, N)
    return rect_l, rect_t, rect_r, rect_b


def nearest_rect(pyramid_rects_l, pyramid_rects_t, pyramid_rects_r, pyramid_rects_b, px, py):
    # ys = []
    # xs = []
    # for i in range(len(pyramid_rects_l)):
    #     x, y = nearest_point(pyramid_rects_l[i], pyramid_rects_t[i], pyramid_rects_r[i], pyramid_rects_b[i], px, py)
    #     ys.append(y)
    #     xs.append(x)
    # ys = torch.tensor(ys)
    # xs = torch.tensor(xs)

    xs, ys = px.clone(), py.clone()
    xs = torch.where(xs < pyramid_rects_l, pyramid_rects_l, xs)
    xs = torch.where(xs > pyramid_rects_r, pyramid_rects_r, xs)
    ys = torch.where(ys < pyramid_rects_t, pyramid_rects_t, ys)
    ys = torch.where(ys > pyramid_rects_b, pyramid_rects_b, ys)
    dists = torch.sqrt(torch.square(xs - px) + torch.square(ys - py))
    idx = torch.argmin(dists, dim=-1)

    return idx


def tiled_pyramid_to_image_vectorized(pyramid_rects_l, pyramid_rects_t, pyramid_rects_r, pyramid_rects_b, rect_l, rect_t, rect_r, rect_b):
    pyramid_down_iter = nearest_rect(pyramid_rects_l, pyramid_rects_t, pyramid_rects_r, pyramid_rects_b, px=(rect_l + rect_r) / 2.0, py=(rect_t + rect_b) / 2.0)

    origin_x = pyramid_rects_l[pyramid_down_iter]
    origin_y = pyramid_rects_t[pyramid_down_iter]

    return rect_up(rect_l.squeeze(-1) - origin_x, rect_t.squeeze(-1) - origin_y, rect_r.squeeze(-1) - origin_x, rect_b.squeeze(-1) - origin_y, pyramid_down_iter)


def tiled_pyramid_to_image(pyramid_rects_l, pyramid_rects_t, pyramid_rects_r, pyramid_rects_b, rect_l, rect_t, rect_r, rect_b):
    pyramid_down_iter = nearest_rect(pyramid_rects_l, pyramid_rects_t, pyramid_rects_r, pyramid_rects_b, px=(rect_l + rect_r) / 2.0, py=(rect_t + rect_b) / 2.0)

    origin_x = pyramid_rects_l[pyramid_down_iter]
    origin_y = pyramid_rects_t[pyramid_down_iter]

    return rect_up(rect_l - origin_x, rect_t - origin_y, rect_r - origin_x, rect_b - origin_y, pyramid_down_iter)


def tensor_to_dets(output_tensor, pyramyd_rects, conv_layers_params, adjust_threshold=0.0):
    # scan the final layer and output the positive scoring locations
    for k in range(1):#= 0; k < (long)options.detector_windows.size(); ++k)
        indices = (output_tensor > 0.0).nonzero(as_tuple=True)
        values = output_tensor[indices]
        sorted_positions = torch.argsort(values, dim=-1, descending=True)
        sorted_indices0 = indices[0][sorted_positions].cpu()
        sorted_indices1 = indices[1][sorted_positions].cpu()
        sorted_indices2 = indices[2][sorted_positions]
        sorted_indices3 = indices[3][sorted_positions]
        scores = values[sorted_positions]
        ys = sorted_indices2
        xs = sorted_indices3
        for conv_layer_params in conv_layers_params:
            # conv_layer_params is [stride, padding, kernel_size // 2]
            ys = ys * conv_layer_params[1] - conv_layer_params[3] + conv_layer_params[5]
            xs = xs * conv_layer_params[0] - conv_layer_params[2] + conv_layer_params[4]
        half_wnd = (80 - 1) / 2
        rects_l = xs - half_wnd
        rects_t = ys - half_wnd
        rects_r = xs + half_wnd
        rects_b = ys + half_wnd
        # for i in range(len(scores)):
        #     score = scores[i]
        #     y = sorted_indices2[i]
        #     x = sorted_indices3[i]
        #     for conv_layer_params in conv_layers_params:
        #         # conv_layer_params is [stride, padding, kernel_size // 2]
        #         x = x * conv_layer_params[0] - conv_layer_params[2] + conv_layer_params[4]
        #         y = y * conv_layer_params[1] - conv_layer_params[3] + conv_layer_params[5]
        #     half_wnd = (80 - 1) / 2
        #     rect = [x - half_wnd, y - half_wnd, x + half_wnd, y + half_wnd]
        rects_l = rects_l.cpu()
        rects_t = rects_t.cpu()
        rects_r = rects_r.cpu()
        rects_b = rects_b.cpu()
        scores = scores.cpu()
        if scores.size()[-1] == 0:
            bboxes_l, bboxes_t, bboxes_r, bboxes_b = rects_l, rects_t, rects_r, rects_b
        else:
            #rects = tiled_pyramid_to_image_(pyramyd_rects, rects_l, rects_t, rects_r, rects_b)
            pyramyd_rects_l = []
            pyramyd_rects_t = []
            pyramyd_rects_r = []
            pyramyd_rects_b = []
            for i in range(len(pyramyd_rects)):
                pyramyd_rects_l.append(pyramyd_rects[i].l)
                pyramyd_rects_t.append(pyramyd_rects[i].t)
                pyramyd_rects_r.append(pyramyd_rects[i].r)
                pyramyd_rects_b.append(pyramyd_rects[i].b)
            pyramyd_rects_l = torch.tensor(pyramyd_rects_l, dtype=torch.float32)
            pyramyd_rects_t = torch.tensor(pyramyd_rects_t, dtype=torch.float32)
            pyramyd_rects_r = torch.tensor(pyramyd_rects_r, dtype=torch.float32)
            pyramyd_rects_b = torch.tensor(pyramyd_rects_b, dtype=torch.float32)
            bboxes_l, bboxes_t, bboxes_r, bboxes_b = tiled_pyramid_to_image_vectorized(pyramyd_rects_l, pyramyd_rects_t, pyramyd_rects_r, pyramyd_rects_b, rects_l.unsqueeze(-1),
                                          rects_t.unsqueeze(-1), rects_r.unsqueeze(-1), rects_b.unsqueeze(-1))
            # for i in range(len(scores)):
            #     rect = tiled_pyramid_to_image(pyramyd_rects_l, pyramyd_rects_t, pyramyd_rects_r, pyramyd_rects_b, rects_l[i], rects_t[i], rects_r[i], rects_b[i])
            #     #print(f"l t r b {rect[0]} {rect[1]} {rect[2]} {rect[3]}")
            #
            #     dets_accum.append([scores[i], rect])

            # for r in range(output_tensor.shape[-2]):
            #     for c in range(output_tensor.shape[-1]):
            #         score = output_tensor[0, 0, r, c]
            #         if (score > adjust_threshold):
            #             x, y = c, r
            #             for conv_layer_params in conv_layers_params:
            #                 # conv_layer_params is [stride, padding, kernel_size // 2]
            #                 x = x * conv_layer_params[0] - conv_layer_params[2] + conv_layer_params[4]
            #                 y = y * conv_layer_params[1] - conv_layer_params[3] + conv_layer_params[5]
            #             half_wnd = (80 - 1) / 2
            #             rect = [x - half_wnd, y - half_wnd, x + half_wnd, y + half_wnd]
            #             rect = tiled_pyramid_to_image(pyramyd_rects, rect)
            #             #print(f"l t r b {rect[0]} {rect[1]} {rect[2]} {rect[3]}")
            #
            #             dets_accum.append([score, rect])

    #dets_accum = sorted(dets_accum, key=lambda det: det[0])[::-1]

    return scores, bboxes_l, bboxes_t, bboxes_r, bboxes_b


def overlaps_nms(a_l, a_t, a_r, a_b,  b_l, b_t, b_r, b_b):
    def is_empty(l, t, r, b):
        return (l > r) or (t > b)
    
    def width(l, t, r, b):
        if is_empty(l, t, r, b):
            return 0
        return r - l + 1

    def height(l, t, r, b):
        if is_empty(l, t, r, b):
            return 0
        return b - t + 1

    def area(l, t, r, b):
        return height(l, t, r, b) * width(l, t, r, b)

    def union(a_l, a_t, a_r, a_b,  b_l, b_t, b_r, b_b):
        return min(a_l, b_l), min(a_t, b_t), max(a_r, b_r), max(a_b, b_b)

    def intersect(a_l, a_t, a_r, a_b,  b_l, b_t, b_r, b_b):
        return max(a_l, b_l), max(a_t, b_t), min(a_r, b_r), min(a_b, b_b)

    inner = area(*intersect(a_l, a_t, a_r, a_b,  b_l, b_t, b_r, b_b))
    if (inner == 0):
        return False

    outer = area(*union(a_l, a_t, a_r, a_b,  b_l, b_t, b_r, b_b))
    iou_thresh = 0.338276
    percent_covered_thresh = 1.0
    if ((inner/outer > iou_thresh) or
        (inner/area(a_l, a_t, a_r, a_b) > percent_covered_thresh) or
        (inner/area(b_l, b_t, b_r, b_b) > percent_covered_thresh)):
        return True
    else:
        return False


def to_label(output_tensor, pyramyd_rects, conv_layers_params, adjust_threshold=0.0):
    output_tensor = output_tensor.detach()
    final_dets = []
    for b in range(output_tensor.shape[0]):
        scores, bboxes_l, bboxes_t, bboxes_r, bboxes_b = tensor_to_dets(output_tensor, pyramyd_rects, conv_layers_params, adjust_threshold)
        scores = scores.numpy().astype(np.float)
        bboxes_l = (bboxes_l + 0.5).numpy().astype(np.int32)
        bboxes_t = (bboxes_t + 0.5).numpy().astype(np.int32)
        bboxes_r = (bboxes_r + 0.5).numpy().astype(np.int32)
        bboxes_b = (bboxes_b + 0.5).numpy().astype(np.int32)

        # Do non-max suppression
        b_final_dets = []
        for i in range(len(scores)):
            bbox_l = bboxes_l[i]
            bbox_t = bboxes_t[i]
            bbox_r = bboxes_r[i]
            bbox_b = bboxes_b[i]

            overlaps_any_box_nms = False
            for score, det in b_final_dets:
                if (overlaps_nms(det[0], det[1], det[2], det[3], bbox_l, bbox_t, bbox_r, bbox_b)):
                    overlaps_any_box_nms = True
                    break
            if overlaps_any_box_nms:
                continue

            detection_confidence = scores[i]
            b_final_dets.append([detection_confidence, [bbox_l, bbox_t, bbox_r, bbox_b]])

        final_dets.append(b_final_dets)

    return final_dets


def bilinear_interpolate_torch(im, x, y):
    x0 = torch.floor(x).type(torch.long)
    x1 = x0 + 1

    y0 = torch.floor(y).type(torch.long)
    y1 = y0 + 1

    x0_clipped = torch.clamp(x0, 0, im.shape[-1] - 1)
    x1_clipped = torch.clamp(x1, 0, im.shape[-1] - 1)
    y0_clipped = torch.clamp(y0, 0, im.shape[-2] - 1)
    y1_clipped = torch.clamp(y1, 0, im.shape[-2] - 1)

    Ia = im[:, :, y0_clipped, x0_clipped]
    Ib = im[:, :, y1_clipped, x0_clipped]
    Ic = im[:, :, y0_clipped, x1_clipped]
    Id = im[:, :, y1_clipped, x1_clipped]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    output = ((Ia) * wa) + ((Ib) * wb) + ((Ic) * wc) + ((Id) * wd)

    return output


class rectangle:
    def __init__(self, l, t, r, b):
        self.l = l
        self.t = t
        self.r = r
        self.b = b

    def width(self):
        if (self.is_empty()):
            return 0
        return self.r - self.l + 1

    def height(self):
        if (self.is_empty()):
            return 0
        return self.b - self.t + 1

    def area(self):
        return self.height() * self.width()

    def union(self, rhs):
        return rectangle(
            min(self.l, rhs.l),
            min(self.t, rhs.t),
            max(self.r, rhs.r),
            max(self.b, rhs.b))

    def intersect(self, rhs):
        return rectangle(
            max(self.l, rhs.l),
            max(self.t, rhs.t),
            min(self.r, rhs.r),
            min(self.b, rhs.b))

    def equals(self, rect):
        return (rect.l == self.l and rect.r == self.r and rect.t == self.t and rect.b == self.b)

    def is_empty(self):
        return (self.l > self.r or self.t > self.b)

    def contains(self, x, y):
        if (x < self.l or x > self.r or y < self.t or y > self.b):
            return False
        return True

    def contains(self, rect):
        return self.union(rect).equals(self)


def find_pyramid_down_output_image_size(pyr_N, nr, nc):
    rate = (pyr_N - 1.0) / pyr_N
    nr = int(rate * nr)
    nc = int(rate * nc)
    return nr, nc


def compute_tiled_image_pyramid_details(pyr_N, nr, nc, padding=10, outer_padding=11):
    rects = []
    if (nr * nc == 0):
        pyramid_image_nr = 0
        pyramid_image_nc = 0
        return pyramid_image_nr, pyramid_image_nc, rects

    min_height = 5
    rects.append(rectangle(0, 0, nc - 1, nr - 1))
    # build the whole pyramid
    while True:
        nr, nc = find_pyramid_down_output_image_size(pyr_N, nr, nc)
        if (nr * nc == 0 or nr < min_height):
            break
        rects.append(rectangle(0, 0, nc - 1, nr - 1))

    # figure out output image size
    total_height = 0
    for rect in rects:
        total_height += rect.height() + padding
    total_height -= padding * 2  # don't add unnecessary padding to the very right side.
    height = 0
    prev_width = 0
    for rect in rects:
        # Figure out how far we go on the first column.We go until the next image can
        # fit next to the previous one, which means we can double back for the second
        # column of images.
        if ((rect.width() <= rects[0].width() - prev_width - padding)
                and ((height - rects[0].height()) * 2 >= (total_height - rects[0].height()))):
            break
        height += rect.height() + padding
        prev_width = rect.width()
    height -= padding  # don't add unnecessary padding to the very right side.

    width = rects[0].width()
    pyramid_image_nr = height + outer_padding * 2
    pyramid_image_nc = width + outer_padding * 2

    y = outer_padding
    i = 0
    while (y < height + outer_padding and i < len(rects)):
        rects[i] = rectangle(rects[i].l + outer_padding, rects[i].t + y, rects[i].r + outer_padding, rects[i].b + y)
        assert rectangle(0, 0, pyramid_image_nc - 1, pyramid_image_nr - 1).contains(rects[i])
        y += rects[i].height() + padding
        i += 1
    y -= padding
    while (i < len(rects)):
        rect = rectangle(outer_padding + width - 1, y - 1, outer_padding + width - 1, y - 1).union(
            rectangle(outer_padding + width - 1 - rects[i].r, y - 1 - rects[i].b,
                      outer_padding + width - 1 - rects[i].r, y - 1 - rects[i].b))
        if not rectangle(0, 0, pyramid_image_nc - 1, pyramid_image_nr - 1).contains(rect):
            a = 1
        assert rectangle(0, 0, pyramid_image_nc - 1, pyramid_image_nr - 1).contains(rect)
        # don't keep going on the last row if it would intersect the original image.
        if (not rects[0].intersect(rect).is_empty()):
            break

        rects[i] = rect
        y -= rects[i].height() + padding
        i += 1

    # Delete any extraneous rectangles if we broke out of the above loop early due to
    # intersection with the original image.
    rects = rects[:i]

    return pyramid_image_nr, pyramid_image_nc, rects


def get_estimated_gender(output):
    gender_id = np.argmax(output, axis=-1)
    gender_confidence = output[np.arange(len(output)), gender_id]

    return gender_id.tolist(), gender_confidence.tolist()


def get_estimated_age(output):
    max_confidence_pos = np.argmax(output, axis=-1)
    age_confidence = output[np.arange(len(output)), max_confidence_pos]
    number_of_age_classes = 81
    age = np.sum(np.arange(0, number_of_age_classes) * output, axis=-1) + (0.25 * output[:, 0])
    age = (age + 0.5).astype(np.int32)

    return age.tolist(), age_confidence.tolist()

    # estimated_age = 0.25 * output[0]
    # confidence = output[0]
    #
    # for k in range(1, number_of_age_classes):
    #     estimated_age += k * output[k]
    #     if output[k] > confidence:
    #         confidence = output[k]
    #
    # return int(estimated_age + 0.5), confidence


class InputRgbImage(nn.Module):
    def __init__(self, r=122.781998, g=117.000999, b=104.297997):
        super(InputRgbImage, self).__init__()
        value_to_subtract_tensor = torch.tensor([r, g, b], dtype=torch_float)
        self.register_buffer(name='value_to_subtract_tensor', tensor=value_to_subtract_tensor, persistent=False)

    def forward(self, input):
        x = (input - self.value_to_subtract_tensor) * (1.0 / 256.0)
        x = x.to(dtype=torch_float)
        x = x.permute((0, 3, 1, 2))
        return x


class InputRgbImagePyramyd(InputRgbImage):
    def __init__(self, padding=10, outer_padding=11, r=122.781998, g=117.000999, b=104.297997):
        super(InputRgbImagePyramyd, self).__init__(r=r, g=g, b=b)
        self.padding = padding
        self.outer_padding = outer_padding

    def forward(self, input):
        assert input.size()[0] == 1, "batch sizes greater than 1 aren't supported yet"

        x = super(InputRgbImagePyramyd, self).forward(input)

        height, width = x.size()[-2:]
        pyramid_image_nr, pyramid_image_nc, rects = compute_tiled_image_pyramid_details(pyr_N=6, nr=height, nc=width,
                                                                                        padding=self.padding, outer_padding=self.outer_padding)
        pyramid_image = torch.zeros((1, 3, pyramid_image_nr, pyramid_image_nc), dtype=torch_float, device=input.device)

        pyramid_tile0 = x

        for level in range(len(rects)):
            if level == 0:
                pyramid_tile = pyramid_tile0
            else:
                # pyramid_tile = nnf.interpolate(pyramid_tile0, size=(rects[level].height(), rects[level].width()), mode='bilinear', align_corners=False)

                width = pyramid_tile0.shape[-1]
                height = pyramid_tile0.shape[-2]
                new_width = rects[level].width()
                new_height = rects[level].height()
                min_face_in_pixels = 10
                if (new_width < min_face_in_pixels) or (new_height < min_face_in_pixels):
                    #print(f'skipping levels as one of rect dims is < 4 pixels')
                    break
                xs = torch.arange(0., width - 1 + 0.000001, (width - 1) / (new_width - 1), device=pyramid_tile0.device)
                xs = torch.reshape(xs, (1, -1))
                ys = torch.arange(0., height - 1 + 0.000001, (height - 1) / (new_height - 1), device=pyramid_tile0.device)
                ys = torch.reshape(ys, (-1, 1))

                pyramid_tile = bilinear_interpolate_torch(pyramid_tile0, x=xs, y=ys)

            pyramid_image[:, :, rects[level].t:rects[level].b + 1, rects[level].l:rects[level].r + 1] = pyramid_tile
            pyramid_tile0 = pyramid_tile

        return pyramid_image, rects


class AddPrev(nn.Module):
    def __init__(self):
        super(AddPrev, self).__init__()

    def forward(self, input0, input1):
        return input0 + input1


class Multiply(nn.Module):
    def __init__(self, factor):
        super(Multiply, self).__init__()
        self.factor = factor

    def forward(self, input):
        return input * self.factor
