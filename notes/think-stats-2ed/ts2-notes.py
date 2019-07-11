# create a DataFrame
import nsfg
def ReadFemPreg(dct_file='2002FemPreg.dct',
                dct_file='2002FemPreg.dat.gz'):
    dct = thinkstats2.ReadStataDct(dct_file)
    df = dct.ReadFixedWidth(dct_file, compression='gzip')
    CleanFemPreg(df)
    return df

def CleanFemPreg(df):
    df.agepreg /= 100.0     # agepreg used centiyears -> convert to years

    na_vals = [97, 98, 99] # values of unswared
    df.birthwgt_lb.replace(na_vals, np.nan, inplace=True)
    df.birthwgt_oz.replace(na_vals, np.nan, inplace=True)

    df['totalwgt_lb'] = df.birthwgt_lb + df.birthwgt_oz / 16.0


df = nsfg.ReadFemPreg()

df.outcome.value_counts().sort_index()  # a Series / sort by index

# clean a error value
df.birthwgt_lb[df.birthwgt_lb > 20] = np.nan

# process to collect the pregnancy data for each respondent
def MakePregMap(df):
    d = defaultdict(list)
    for index, caseid in df.caseid.iteritems():
        d[caseid].append(index)
    return d

preg_map = MakePregMap(df)
caseid = 10229
indices = preg_map[caseid]      # indices for pregnancies to respondent caseid=10229
df.outcome[indices].values
