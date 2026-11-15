import numpy as np


def find_closest_index(mjd: float, thar_times: np.ndarray) -> tuple[int, float]:
    """Finds the index of the closest MJD value in a sorted numpy array.

    Parameters:
        mjd (float): Target MJD value (e.g., science frame timestamp).
        thar_times (np.ndarray): Sorted 1D array of MJD timestamps.

    Returns:
        tuple[int, float]: (best_match_index, diff_seconds)
            - best_match_index: Index of the closest element.
            - diff_seconds: Absolute difference in seconds.
    """
    # Convert list to numpy array if it isn't one already
    thar_times = np.asarray(thar_times)

    # Find the insertion index to maintain sorted order (binary search)
    idx = np.searchsorted(thar_times, mjd)

    # Edge cases: mjd is outside the range of thar_times
    if idx == 0:
        best_idx = 0
    elif idx == len(thar_times):
        best_idx = len(thar_times) - 1
    else:
        # mjd falls between idx - 1 and idx; check which one is closer
        left_diff = mjd - thar_times[idx - 1]
        right_diff = thar_times[idx] - mjd

        if left_diff <= right_diff:
            best_idx = idx - 1
        else:
            best_idx = idx

    # Calculate absolute difference in MJD days and convert to seconds (1 day = 86400 seconds)
    mjd_diff = abs(thar_times[best_idx] - mjd)
    diff_seconds = mjd_diff * 86400.0

    return best_idx, diff_seconds


# Example usage
if __name__ == "__main__":
    # Sorted array of ThAr MJD timestamps
    thar_times = np.array(
        [
            61306.500000,
            61306.550000,
            61306.560000,
            61306.570000,
            61306.600000,
        ]
    )

    science_mjd = 61306.56101852

    best_match_index, diff_seconds = find_closest_index(science_mjd, thar_times)

    print(f"Target Science MJD : {science_mjd}")
    print(f"Closest ThAr MJD    : {thar_times[best_match_index]}")
    print(f"Best match index    : {best_match_index}")
    print(f"Time difference     : {diff_seconds:.2f} seconds")