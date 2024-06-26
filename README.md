# Segment Phasing Focal-plane Env

<p align="center">
    <img style="width:200px" src="https://raw.githubusercontent.com/jcranney/segment-phasing-fp-env/main/segment_phasing_fp_env/images/loop_closed_fake.GIF"
        alt="loop closing fake"/>
</p>

<table>
    <tbody>
        <tr>
            <td>Action Space</td>
            <td>Box(6)</td>
        </tr>
        <tr>
            <td>Observation Shape</td>
            <td>(20, 20)</td>
        </tr>
        <tr>
            <td>Observation High</td>
            <td>65535</td>
        </tr>
        <tr>
            <td>Observation Low</td>
            <td>0</td>
        </tr>
        <tr>
            <td>Import</td>
            <td>import segment_phasing_fp_env  # noqa<br/>gymnasium.make("SegmentPhasingFP-v0")</td>
        </tr>
    </tbody>
</table>

## Description

GMT focal-plane segment phasing as a Farama Gymnasium environment.

### Installation

```bash
git clone https://github.com/jcranney/segment_phasing_fp_env
cd segment_phasing_fp_env
pip install .
```

### Usage

1. Play it by running

```bash
python -m segment_phasing_fp_env
```

The above "playing" mode is mostly useless, but if it runs then everything has been installed correctly. Press `space` to drive the 1st segment up, which should have a notable effect on the focal-plane image. 

If the above doesn't work due to a `GLIBCXX` error, try:
```bash
conda install -c conda-forge libstdcxx-ng=12
```
(see [here](https://github.com/microsoft/DeepSpeed/issues/2886))

2. Import it to train your RL model

```python
import gymnasium
import segment_phasing_fp_env
env = gymnasium.make("SegmentPhasingFP-v0")
```

The package relies on ```import``` side-effects to register the environment
name so, even though the package is never explicitly used, its import is
necessary to access the environment.

## Action Space

`SegmentPhasingFP-v0` has the action space `Box(low=-np.inf, high=np.inf, shape=(6,))`. Each element corresponds to a change in the segment piston command, in units of radians. Note that the system is blind to global piston, so there are only 6 degrees of freedom to actuate. These are the global-piston-removed segment piston modes for the outer six segments:

<p align="center">
    <img src="https://raw.githubusercontent.com/jcranney/segment-phasing-fp-env/main/segment_phasing_fp_env/images/modes.png"
        alt="modes.png"/>
</p>


## Observation Space

The observation is the noisy and slightly undersampled OIWFS image, `Box(low=0, high=65536, shape=(20, 20), dtype=np.uint16)`.

### Rewards

You get `+{strehl_ratio}` every time you make it another step without dipping below the Strehl minimum of 10%.

## Version History

- v0: initial version release
