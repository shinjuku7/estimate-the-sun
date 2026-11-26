#!/usr/bin/env python3
"""Blend the two predictions and set fully dark intervals to zero."""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
warnings.filterwarnings('ignore')
import features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--mine', required=True)
    ap.add_argument('--other', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--level-match', action='store_true')
    a = ap.parse_args()

    test = pd.read_csv(os.path.join(a.data, 'test.csv'), usecols=['id', 'datetime_utc'])
    mine = pd.read_csv(a.mine)
    other = pd.read_csv(a.other)
    for name, d in (('mine', mine), ('other', other)):
        assert len(d) == len(test) and set(d.id) == set(test.id), f'{name}: ids do not match test.csv'
        assert d.generation_mw.notna().all(), f'{name}: missing values'
    d = test.merge(mine, on='id').merge(other, on='id', suffixes=('_mine', '_other'))

    F = features.build(features.load(a.data))
    sun = F.loc[F.is_train == 0, ['id', 'sun_any']]
    d = d.merge(sun, on='id', how='left')
    assert d.sun_any.notna().all()

    blend = 0.5 * (d.generation_mw_mine.values + d.generation_mw_other.values)
    blend[d.sun_any.values <= 0] = 0.0
    if a.level_match:
        month = pd.to_datetime(d.datetime_utc, utc=True).dt.month.values
        tot_mine = pd.Series(d.generation_mw_mine.values).groupby(month).transform('sum').values
        tot_blend = pd.Series(blend).groupby(month).transform('sum').values
        blend = blend * np.where(tot_blend > 0, tot_mine / tot_blend, 1.0)

    out = test[['id']].merge(pd.DataFrame({'id': d.id, 'generation_mw': np.round(np.clip(blend, 0, None), 4)}),
                             on='id', how='left')
    assert out.generation_mw.notna().all()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    out.to_csv(a.out, index=False, lineterminator="\n")
    print(f'wrote {a.out}: mean {out.generation_mw.mean():.1f} MW '
          f'(mine {d.generation_mw_mine.mean():.1f}, other {d.generation_mw_other.mean():.1f}), '
          f'level_match={a.level_match}')


if __name__ == '__main__':
    main()
