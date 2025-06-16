import torch
import numpy as np
from skimage import color


def match_and_blend_colors(source_chunk: torch.Tensor, reference_image: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Matches the color of a source video chunk to a reference image and blends with the original.

    Args:
        source_chunk (torch.Tensor): The video chunk to be color-corrected (B, C, T, H, W) in range [-1, 1].
                                     Assumes B=1 (batch size of 1).
        reference_image (torch.Tensor): The reference image (B, C, 1, H, W) in range [-1, 1].
                                        Assumes B=1 and T=1 (single reference frame).
        strength (float): The strength of the color correction (0.0 to 1.0).
                          0.0 means no correction, 1.0 means full correction.

    Returns:
        torch.Tensor: The color-corrected and blended video chunk.
    """
    if strength == 0.0:
        return source_chunk

    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"Strength must be between 0.0 and 1.0, got {strength}")

    device = source_chunk.device
    dtype = source_chunk.dtype

    # Squeeze batch dimension, permute to T, H, W, C for skimage
    # Source: (1, C, T, H, W) -> (T, H, W, C)
    source_np = source_chunk.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    # Reference: (1, C, 1, H, W) -> (H, W, C)
    ref_np = reference_image.squeeze(0).squeeze(1).permute(1, 2, 0).cpu().numpy() # Squeeze T dimension as well

    # Normalize from [-1, 1] to [0, 1] for skimage
    source_np_01 = (source_np + 1.0) / 2.0
    ref_np_01 = (ref_np + 1.0) / 2.0

    # Clip to ensure values are strictly in [0, 1] after potential float precision issues
    source_np_01 = np.clip(source_np_01, 0.0, 1.0)
    ref_np_01 = np.clip(ref_np_01, 0.0, 1.0)

    # Convert reference to Lab
    try:
        ref_lab = color.rgb2lab(ref_np_01)
    except ValueError as e:
        # Handle potential errors if image data is not valid for conversion
        print(f"Warning: Could not convert reference image to Lab: {e}. Skipping color correction for this chunk.")
        return source_chunk


    corrected_frames_np_01 = []
    for i in range(source_np_01.shape[0]): # Iterate over time (T)
        source_frame_rgb_01 = source_np_01[i]
        
        try:
            source_lab = color.rgb2lab(source_frame_rgb_01)
        except ValueError as e:
            print(f"Warning: Could not convert source frame {i} to Lab: {e}. Using original frame.")
            corrected_frames_np_01.append(source_frame_rgb_01)
            continue

        corrected_lab_frame = source_lab.copy()

        # Perform color transfer for L, a, b channels
        for j in range(3): # L, a, b
            mean_src, std_src = source_lab[:, :, j].mean(), source_lab[:, :, j].std()
            mean_ref, std_ref = ref_lab[:, :, j].mean(), ref_lab[:, :, j].std()

            # Avoid division by zero if std_src is 0
            if std_src == 0:
                # If source channel has no variation, keep it as is, but shift by reference mean
                # This case is debatable, could also just copy source or target mean.
                # Shifting by target mean helps if source is flat but target isn't.
                corrected_lab_frame[:, :, j] = mean_ref 
            else:
                corrected_lab_frame[:, :, j] = (corrected_lab_frame[:, :, j] - mean_src) * (std_ref / std_src) + mean_ref
        
        try:
            fully_corrected_frame_rgb_01 = color.lab2rgb(corrected_lab_frame)
        except ValueError as e:
            print(f"Warning: Could not convert corrected frame {i} back to RGB: {e}. Using original frame.")
            corrected_frames_np_01.append(source_frame_rgb_01)
            continue
            
        # Clip again after lab2rgb as it can go slightly out of [0,1]
        fully_corrected_frame_rgb_01 = np.clip(fully_corrected_frame_rgb_01, 0.0, 1.0)

        # Blend with original source frame (in [0,1] RGB)
        blended_frame_rgb_01 = (1 - strength) * source_frame_rgb_01 + strength * fully_corrected_frame_rgb_01
        corrected_frames_np_01.append(blended_frame_rgb_01)

    corrected_chunk_np_01 = np.stack(corrected_frames_np_01, axis=0)

    # Convert back to [-1, 1]
    corrected_chunk_np_minus1_1 = (corrected_chunk_np_01 * 2.0) - 1.0

    # Permute back to (C, T, H, W), add batch dim, and convert to original torch.Tensor type and device
    # (T, H, W, C) -> (C, T, H, W)
    corrected_chunk_tensor = torch.from_numpy(corrected_chunk_np_minus1_1).permute(3, 0, 1, 2).unsqueeze(0)
    corrected_chunk_tensor = corrected_chunk_tensor.contiguous() # Ensure contiguous memory layout
    return corrected_chunk_tensor.to(device=device, dtype=dtype)

if __name__ == '__main__':
    # Basic test case
    print("Running basic test for match_and_blend_colors...")
    B, C, T, H, W = 1, 3, 5, 64, 64 # Batch, Channels, Time, Height, Width
    
    # Create a dummy source chunk (e.g., mostly red)
    source = torch.ones((B, C, T, H, W), dtype=torch.float32) * -1.0 # Start with -1
    source[:, 0, :, :, :] = 0.8 # Red channel high
    source[:, 1, :, :, :] = -0.5 # Green channel low
    source[:, 2, :, :, :] = -0.8 # Blue channel low

    # Create a dummy reference image (e.g., mostly blue)
    reference = torch.ones((B, C, 1, H, W), dtype=torch.float32) * -1.0
    reference[:, 0, :, :, :] = -0.8 # Red channel low
    reference[:, 1, :, :, :] = -0.5 # Green channel low
    reference[:, 2, :, :, :] = 0.8 # Blue channel high

    strength_test = 0.75
    
    print(f"Source shape: {source.shape}, Reference shape: {reference.shape}, Strength: {strength_test}")
    
    corrected_tensor = match_and_blend_colors(source.clone(), reference.clone(), strength_test)
    print(f"Corrected tensor shape: {corrected_tensor.shape}")
    print(f"Corrected tensor dtype: {corrected_tensor.dtype}")

    # Check a few values (example)
    print(f"Original source (R channel, 1st frame, 1st pixel): {source[0, 0, 0, 0, 0].item():.3f}")
    print(f"Corrected (R channel, 1st frame, 1st pixel): {corrected_tensor[0, 0, 0, 0, 0].item():.3f}")
    
    print(f"Original source (B channel, 1st frame, 1st pixel): {source[0, 2, 0, 0, 0].item():.3f}")
    print(f"Corrected (B channel, 1st frame, 1st pixel): {corrected_tensor[0, 2, 0, 0, 0].item():.3f}")

    # Test strength = 0
    corrected_strength_0 = match_and_blend_colors(source.clone(), reference.clone(), 0.0)
    assert torch.allclose(source, corrected_strength_0), "Strength 0.0 did not return original tensor"
    print("Strength 0.0 test passed.")

    # Test strength = 1 (should be different from source, closer to reference color profile)
    corrected_strength_1 = match_and_blend_colors(source.clone(), reference.clone(), 1.0)
    assert not torch.allclose(source, corrected_strength_1), "Strength 1.0 returned original tensor (unexpected for different images)"
    print("Strength 1.0 test produced a different tensor (as expected).")
    
    print("Basic test completed.")