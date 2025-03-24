# -*- coding: utf-8 -*-
# author Titouan Poquillon
# 
# DEDICATED MODUL FOR STATISTICS AND PLOTING

from src.external import *
from skimage.filters.thresholding import gaussian
from scipy import stats
import matplotlib.patches as patches
import seaborn as sns
class DESCRIPTOR_TYPE(Enum):
    NUMERIC = 1
    CATEGORY = 2
COLORS_LIGHT = list(matplotlib.colormaps["tab20"].colors)[:-1] #+ list(matplotlib.colormaps["Set2").colors)
COLORS_STRONG = list(matplotlib.colormaps["tab10"].colors)[::-1]
COLORS_DARK = list(matplotlib.colormaps["Dark2"].colors)

def thresh2Val(val, thresh_dict = {1:5,0.05:10,0.01:20,0.001:40}):
    """convert list of values to new list of values according to a threshold:value dictionary

    Args:
        val (np.ndarray)
        thresh_dict (dict, optional):  threshold 2 value dictionary. Defaults to {1:5,0.05:10,0.01:20,0.001:40}.

    Returns:
        np.ndarray: array with converted values
    """
    thresh = list(thresh_dict.keys())
    go_smaller = thresh[0]>thresh[1]
    out = np.ones(val.shape,dtype=val.dtype)
    out[...] = thresh_dict[thresh[0]] if go_smaller else  thresh_dict[thresh[-1]]
    for t in (thresh[1:] if go_smaller else thresh[-2::-1]):
        if go_smaller:
            out[val<t]=thresh_dict[t]
        else:
            out[val>t]=thresh_dict[t]
    return out
def progresSpeedMeasurement(t_init, n, n_max):
    """measure the speed of a process ad generate a message accordingly

    Args:
        t_init (int): time at wrich the process started
        n (int): curent iteration of the process
        n_max (int): number of iteration for the process to end

    Returns:
        str: message formated as folow "<<time spent>> : <<time remaining>>  << speed>>"
    """
    progress = n/n_max
    t_past = time.time()-t_init
    t_remain = max(t_past * (1-progress)/progress,0)
    speed = n/t_past if t_past!=0 else 0
    if speed < 1:
        speed_msg = ("%f sec/iter")%(1/speed if speed!=0 else 0)
    else:
        speed_msg = ("%f iter/sec ")%(speed)
    d_past = list(time.localtime(t_past)[3:6])
    d_remain = list(time.localtime(t_remain)[3:6])
    d_past[0]-=1
    d_remain[0]-=1
    msg = "[%dh%dm%ds>%dh%dm%ds, %s]"%(*d_past,*d_remain,speed_msg)
    return msg
def smartIndex(index_):
    """convert dataFrame index or multyIndex to a nice readable string

    Args:
        index_ (pandas.Index)

    Returns:
        list(str): list of string
    """
    label = [" ".join([str(ind) for ind in inds if str(ind) !="-"]) if type(inds) is tuple else (str(inds) if str(inds) !="-" else "") for inds in index_]
    label = [ind if ind!="" else "NT" for ind in label]
    return label
def merge_pdfs(input_files: list, page_range: tuple, output_file: str, bookmark: bool = True):
    """
    Merge a list of PDF files and save the combined result into the `output_file`.
    `page_range` to select a range of pages (behaving like Python's range() function) from the input files
        e.g (0,2) -> First 2 pages 
        e.g (0,6,2) -> pages 1,3,5
    bookmark -> add bookmarks to the output file to navigate directly to the input file section within the output file.
    """
    # strict = False -> To ignore PdfReadError - Illegal Character error
    merger = PdfFileMerger(strict=False)
    for input_file in input_files:
        bookmark_name = Path(input_file).name.split(".")[0].split("__")[0] if bookmark else None
        # pages To control which pages are appended from a particular file.
        merger.append(fileobj=open(input_file, 'rb'), pages=page_range, import_bookmarks=False, bookmark=bookmark_name)
    # Insert the pdf at specific page
    merger.write(fileobj=open(output_file, 'wb'))
    merger.close()
def smartFloat2Str(val,size=4):
    """convert float to string, eventually using scientific notation

    Args:
        val (float): input float
        size (int, optional): number of wanted digit. Defaults to 4.

    Returns:
        str: float string
    """
    val_sc = '%.1E'%Decimal(float(val))
    to_str =  str(val)[:min(size,len(str(val)))] if int(val_sc[-2:])<(size-1) else val_sc
    if to_str[-1] == ".": to_str=to_str[:-1]
    return to_str
def force_numeric_or_NaN(df:pd.DataFrame):
    out = df.copy()
    columns = df.columns
    for col in columns:
        is_numeric = df[col].dtype.kind in 'biufc'
        if not is_numeric:
            c = pd.to_numeric(df[col],"coerce")
            is_numeric = c.dtype.kind in 'biufc'
            
            out[col] = c if is_numeric else np.nan
            
    return out


class Descriptor():
    """Class for descriptor display in pysimplegui.Listbox. and usage. Decriptor can be filtered and belong to a cluster of descriptors
    """
    def __init__(self,name,dec_type = DESCRIPTOR_TYPE.NUMERIC):
        self.name = name
        self.flags = []
        self.origine = "Measure"
        self.type = dec_type
        self.cluster = {"C0":[self]}
        self.str_ = " ".join(name.split("_")).capitalize()
        self.ignore = False
    def rename(self,new_name):
        self.name = new_name
        self.origine = "Measure"
        self.setCluster(self.cluster)
    def __str__(self) -> str:
        return self.str_
    def getKey(self):
        return self.name
    def clustId(self, replace_C0=None):
        key = list(self.cluster.keys())[0]
        return key if key != "C0" else (self.name if replace_C0 is None else replace_C0)
    def keep(self, allow_category = False):
        return not self.ignore and(allow_category or self.type is not DESCRIPTOR_TYPE.CATEGORY)
    def setIgnore(self,bool_val):
        self.ignore=bool_val
    def setCluster(self,cluster):
        self.cluster = cluster
        fammily = list(cluster.keys())[0]
        self.str_ = " ".join(self.name.split("_")).capitalize() + (" - %s "%fammily if fammily!="C0" else "")
    def __eq__(self,str_):
        return self.name == str_

def autoAnalysis(data, ref, c_p, trial, descriptors):
    """generate an AutoAnalysis PDF with a "Take home message" showing a similarity score between trial and reference (negativ control),
        the position of the trial on each negativ control / positiv control scale and a radar Plot with the 15 best descriptors

    Args:
        ref (list[index element]): the reference(Negativ Control)
        c_p (list[index element]): Negativ Controls, may be empty
        trial (list[index element]): Trials, if empty all condition such as in self.data.index.unique are choosen
    """
    desc = list(descriptors.keys())
    out_ref = data[desc].loc[ref]
    data = data.copy()

    if trial is not None and len(trial)>0:
        t_keys = trial
        data = data.loc[list(set(t_keys+c_p))][desc]
    else:
        data=data[desc]
        t_keys = list(data.index.unique())

    
    lda_data = data.copy()
    ldaTvRscore = 2*lda_data.groupby(lda_data.index.names).apply(LDAScoreTvR,out_ref[desc],10) -1
    ldaTvRscore[ldaTvRscore.isna()+(ldaTvRscore<0)] = 0
    lda_cp_dict = pd.DataFrame(LDAcpScaleMaster(lda_data.loc[t_keys],lda_data.loc[c_p],out_ref[desc]), index=ldaTvRscore.index)
    if len(lda_cp_dict.columns)>0 and type(lda_cp_dict.columns[0]) is tuple and len(lda_cp_dict.columns[0])==1:
        lda_cp_dict.columns = [el[0] for el in lda_cp_dict.columns]
    summary = data.groupby(data.index.names).apply(SSMD,out_ref).droplevel(-1)
    sinif = data.groupby(data.index.names).apply(mannWhitney,out_ref).droplevel(-1)
    sinif = sinif.groupby(sinif.index.names).apply(benjaminiHochbergAdjust,0.05)

    ds = ldaTvRscore ; bo = summary; pcs = lda_cp_dict; 
    index = ldaTvRscore.index
    temp_pdf = tempfile.mkstemp(suffix='.pdf',prefix="Analyse_report")
    pdf_list = []
    y=200
    t0 = time.time()
    for i,el in enumerate(index):
        k=i+1
        msg  = progresSpeedMeasurement(t0,k,len(index))
        # self.window.write_event_value("PbUpdate", (int(200*k/len(index)), "Analysis %d/%d "%(k, len(index)),msg))
        try:
            subsinif = summaryCorrelationReduction(descriptors,sinif.loc[[el]].T,"absMin", return_first_el_desc=True)
            best_desc = subsinif.sort_values(by=subsinif.columns[0]).iloc[:min(15,len(subsinif)),:].reset_index()["first_el_desc"].values

            fig = autoPlot(ldaTvRscore.loc[[el]], bo.loc[[el]+[cp_ for cp_ in pcs.columns if cp_!= el]][best_desc], pcs.loc[[el]],ref,sinif.loc[[el]+[cp_ for cp_ in pcs.columns if cp_!= el]][best_desc])

            temp = tempfile.mkstemp(suffix='.pdf',prefix="%s__"%smartIndex([el])[0])
            plt.savefig(temp[1],format="pdf", dpi=50)
            pdf_list.append(temp[1])
            plt.close()
        except:
            pass
    merge_pdfs(pdf_list,(0,1),temp_pdf[1],)
    return temp_pdf[1]

def autoReplicateRadarEvent(data, ref,c_p,trial=[],descriptors:dict={},lim=6,thresh=0.05,group_by=[] ):
    
    desc  = list(descriptors.keys())
    out_ref = data[desc].loc[ref]
    data = data.copy()

    if trial is not None and len(trial)>0:
        t_keys = [el.getKey() if len(el.getKey())>1 else el.getKey()[0] for el in trial]
        data = data.loc[list(set(t_keys+c_p))][desc]
    else:
        data=data[desc]
        t_keys = list(data.index.unique())

    summary = data.groupby(data.index.names).apply(SSMD,out_ref).droplevel(-1).T
    sinif = data.groupby(data.index.names).apply(mannWhitney,out_ref).droplevel(-1)
    sinif = sinif.groupby(sinif.index.names).apply(benjaminiHochbergAdjust,0.05).T
    summary = summaryCorrelationReduction(descriptors,summary,group_by_object = True)
    sinif = summaryCorrelationReduction(descriptors,sinif,"absMin",group_by_object = True)
    sinif.values[...] = thresh2Val(sinif.values,thresh_dict = {1:3,thresh:15,thresh/5:25,thresh/50:40})
    index = summary.columns
    temp_pdf = tempfile.mkstemp(suffix='.pdf',prefix="Analyse_report")
    pdf_list = []
    y=200
    t0 = time.time()
    grouping_data = (len(group_by)>0 and group_by[0] in index.names)
    enum_index = index.copy() if not grouping_data else index.get_level_values(group_by[0]).unique()
    for i,el in enumerate(enum_index):
        k=i+1
        try:
            index_of_data = ([el] if not grouping_data else list(index[index.get_level_values(group_by[0])==el]))+c_p
            index_of_data = list(set(index_of_data))
            if ref[0] in index_of_data:
                continue
            fig=spiderplotMulty(summary[index_of_data],sinif[index_of_data],control_name=str(ref[0]),shiny=True,lim=(-lim,lim),thresh=thresh,sort_val=False)[0]
            temp = tempfile.mkstemp(suffix='.pdf',prefix="%s__"%smartIndex([el])[0])
            plt.savefig(temp[1],format="pdf", dpi=50)
            pdf_list.append(temp[1])
            plt.close()
        except:
            pass
    merge_pdfs(pdf_list,(0,1),temp_pdf[1],)
    return temp_pdf[1]

def indexAsStringList(index:pd.Index)->list:
    return ["__".join(map(str,el)) for el in index] if len(index.names)>1 else list(index.astype("string"))

def getFilteredData(data, **kwargs):
    """filter data according to args 
    Args:
        data (pandas.DataFrame): dataframe to be filtered
        **kwargs: key / value or key / list(value) paires where keys are columns of data

    Returns:
        pandas.DataFrame: a filtered dataframe
    """
    keys = list(kwargs.keys())
    filtered_data = data.copy()
    for key in keys:
        filter = kwargs[key]
        if filter is not None:
            if type(filter) == list:
                filtered_data = filtered_data[filtered_data[key].isin(filter)]
            else:
                filtered_data = filtered_data[filtered_data[key]==filter]
    return filtered_data

def summaryCorrelationReduction(descriptors, summary,reduction_type = "absMax",  group_by_object = False, return_first_el_desc = False):
    """
    Reduced a summary test dataframe (condition x descriptors, tested versus reference x descriptors ) by only keeping the best value for each group of same
        cluster descriptors for each condition
    """   
    s_index = summary.index

    cluster = [descriptors[desc].clustId() for desc in s_index]
    summary["Cluster"]=cluster
    if  group_by_object:
        summary["Obj"] = [el.split("_")[0] for el in s_index]
        summary.set_index('Obj', append=True, inplace=True)
    
    summary.set_index('Cluster', append=True, inplace=True)
    
    order = np.argsort(abs(summary),axis=0)
    clust_order = order.groupby("Cluster" if not group_by_object else ["Cluster","Obj"]).apply(max if reduction_type == "absMax" else min)
    s_index = order.sort_values(by=order.columns[0]).droplevel("Cluster").index
    desc_index = s_index[clust_order.T.values[0]]
    new_summary = summary.groupby("Cluster" if not group_by_object else ["Cluster","Obj"]).apply(absMax if reduction_type == "absMax" else absMin)

    new_summary.index = new_summary.index.droplevel(-1)
    if group_by_object:
        new_summary = new_summary.reset_index()
        new_summary.index = [obj+"_"+clust if "_" not in clust else clust for obj,clust in zip(new_summary["Obj"],new_summary["Cluster"])]
        new_summary = new_summary.sort_index(axis=1)
        new_summary = new_summary.drop("Obj",axis=1)
        new_summary = new_summary.drop("Cluster",axis=1)
    if return_first_el_desc:
        new_summary["first_el_desc"] = desc_index
        new_summary.set_index("first_el_desc",append=True,inplace=True)
    return new_summary

def tTest(data1,data2,keys = None, pval_only=True):
    """Perform an independent tTest on data

    Args:
        data1 (pandas.DataFrame): first dataframe
        data2 (pandas.DataFrame): second dataframe
        keys (str or list, optional): _description_. Defaults to None.

    Returns:
        array-like, array-like: T-value(s) and P-value(s)
    """
    if keys is not None:
        stat, pval = ttest_ind(data1[[keys]],data2[[keys]],equal_var=False,nan_policy="omit")
        col = keys
    else:
        stat, pval = ttest_ind(data1,data2,equal_var=False,nan_policy="omit")
        col = data1.columns

    stat = pd.DataFrame([np.array(stat)],columns=col )
    pval = pd.DataFrame([np.array(pval)],columns=col)
    if pval_only:
        return pval
    return stat, pval

def mannWhitney(data1,data2,keys = None, pval_only=True, less = None, greater = None):
    """Perform The Mann-Whitney U test on data

    Args:
        data1 (pandas.DataFrame): first dataframe
        data2 (pandas.DataFrame): second dataframe
        keys (str or list, optional): _description_. Defaults to None.

    Returns:
        array-like, array-like: U-value(s) and P-value(s)
    """
    if less is not None or greater is not None:

        if less is not None:
            stat_less, pval_less = mannwhitneyu(data1[less],data2[less],nan_policy="omit", alternative="less")
            stat, pval, col = stat_less, pval_less, less
        if greater is not None:
            stat_greater, pval_greater = mannwhitneyu(data1[greater],data2[greater],nan_policy="omit", alternative="greater")            
            stat, pval, col = stat_less, pval_less, greater
        if less is not None and greater is not None:
            stat = np.concatenate([stat_less,stat_greater])
            pval = np.concatenate([pval_less,pval_greater])
            col = less+greater
    elif keys is not None:
        stat, pval = mannwhitneyu(data1[[keys]],data2[[keys]],nan_policy="omit")
        col = keys
    else:
        stat, pval = mannwhitneyu(data1,data2,nan_policy="omit")
        col = data1.columns

    stat = pd.DataFrame([np.array(stat)],columns=col )
    pval = pd.DataFrame([np.array(pval)],columns=col)
    if pval_only:
        return pval
    return stat, pval

def benjaminiHochberg(pval,threshold=0.05):
    """control the false discovery rate of a statistical test using the Benjamini-Hochberg Procedure.

    Args:
        pval (pandas.DataFrame): 1d pval dataframe
        threshold (float): false discovery rate threshold

    Returns:
        pandas.DataFrame: 1d boolean dataframe showing wich descriptor ar passing the BH correction test
    """
    n=pval.shape[1]
    sorted_pval = pval.sort_values(by=pval.index[0], axis=1)
    rank = sorted_pval.rank(axis=1)
    BH_crit = (rank/n)*threshold
    last_rank = ((BH_crit>sorted_pval)*rank).max(axis=1).values[0]
    return (rank<=last_rank)[pval.columns] 

def benjaminiHochbergAdjust(pval,threshold=0.05):
    """return "corected" pvals via BH procedure

    Args:
        pval (pd.Dataframe): input pvalues
        threshold (float, optional): Deprecated. Defaults to 0.05.

    Returns:
        pd.Dataframe: "corected" pvals_
    """
    n=pval.shape[1]
    sorted_pval = pval.sort_values(by=pval.index[0], axis=1)
    rank = sorted_pval.rank(axis=1)
    sorted_pval = sorted_pval*n/rank
    return sorted_pval[pval.columns] 

def SSMD(data1,data2):
    """Measure SSMD between 2 dataframe

    Args:
        data1 (pandas.dataframe): first dataframe
        data2 (pandas.dataframe): second dataframe

    Returns:
        pandas.dataframe: ssmd dataframe
    """
    ssmd = (data1.mean() - data2.mean())/(data1.var()+data2.var())**0.5
    ssmd = pd.DataFrame(ssmd, columns=["ssmd"]).T
    return ssmd

def PCA(data, n=3):
    """Performe PCA on data

    Args:
        data (pandas.Dataframe): each row is a data point, each column a descriptor
        n (int, optional): number of axes for the pca.

    Returns:
        tuple: 
            - pandas.Dataframe : datapoints projected in the PCA 
            - pandas.Dataframe : descriptor correlation with the PCA axes
            - pandas.Dataframe : Explained variance ratio of each PCA axes
    """
    normalized_data=(data-data.mean())/data.std()
    normalized_data[normalized_data.isna()]=0
    pca = sklearn_PCA(n_components=n)
    pca.fit(normalized_data)
    out = pca.transform(normalized_data)    
    out = pd.DataFrame(out, index=data.index, columns=["Ax%d"%(i+1)for i in range(n)])
    out = 2*((out-out.min())/(out.max()-out.min())-0.5)
    comp = pd.concat([normalized_data,out],axis=1).corr()[out.columns].loc[normalized_data.columns].T

    return out, comp, pca.explained_variance_ratio_

def LDA(data, n=3):
    """Performe LDA on data

    Args:
        data (pandas.Dataframe): each row is a data point, each column a descriptor
        n (int, optional): number of axes for the LDA.

    Returns:
        tuple: 
            - pandas.Dataframe : datapoints projected in the LDA 
            - pandas.Dataframe : descriptor correlation with the LDA axes
            - pandas.Dataframe : Explained variance ratio of each LDA axes
    """
    normalized_data=(data-data.mean())/data.std()
    normalized_data[normalized_data.isna()]=0
    lda = sklearn_LDA(n_components=n)
    y=indexAsStringList(data.index)
    lda.fit(normalized_data,y)
    out = lda.transform(normalized_data)
    n = out.shape[1]
    out = pd.DataFrame(out, index=data.index, columns=["Ax%d"%(i+1)for i in range(n)])
    out = 2*((out-out.min())/(out.max()-out.min())-0.5)
    comp = pd.concat([normalized_data,out],axis=1).corr()[out.columns].loc[normalized_data.columns].T

    return out, comp, lda.explained_variance_ratio_

def LDAScore(data, n=3, repeat = 10, return_train_score = False):
    """ Compute a balanced accuracy score via cross validation on a LDA

    Args:
        data (pandas.Dataframe): each row is a data point, each column a descriptor
        n (int, optional): number of axes for the LDA.
        repeat (int, optional): number of cross validation. Defaults to 10.
        return_train_score (bool, optional): if true, also return train balanced accuracy  . Defaults to False.

    Returns:
        float: test cross validation balanced accuracy
    """
    normalized_data=(data-data.mean())/data.std()
    normalized_data[normalized_data.isna()]=0
    
    lda = sklearn_LDA(n_components=n)
    y=indexAsStringList(data.index)
    
    try:
        scores = cross_validate(lda,X=normalized_data,y=y,cv=StratifiedShuffleSplit(repeat,test_size=0.5), scoring="balanced_accuracy",return_train_score=return_train_score)
        mean_accuracy = (scores["test_score"]).mean()
    except Exception as e:
        # print(np.unique(y), "ERREUR\n",e)
        mean_accuracy = np.nan
    return mean_accuracy

def LDAScoreTvR(trial,ref,repeat=10):
    """ Compute a balanced accuracy score via cross validation on a LDA between 2 condition

    Args:
        trial (pandas.Dataframe): each row is a data point, each column a descriptor of trial condition
        ref (pandas.Dataframe): each row is a data point, each column a descriptor of ref condition
        repeat (int, optional): number of cross validation. Defaults to 10.

    Returns:
        float: test cross validation balanced accuracy
    """
    data = pd.concat([trial,ref],axis=0)
    return LDAScore(data, n=1, repeat = repeat)

def LDAcpScaleMaster(trial,control_p,ref):
    """ train a LDA on negativ control and positiv controls and predict trials outcome

    Args:
        trial (pandas.Dataframe): each row is a data point, each column a descriptor of all trials condition
        ref (pandas.Dataframe): each row is a data point, each column a descriptor of ref condition
        control_p (pandas.Dataframe): each row is a data point, each column a descriptor of all positiv control condition

    Returns:
        dict : positive control likelihood for each trial for each positiv control
    """

    data = pd.concat([control_p,ref,trial],axis=0)
    normalized_data=(data-data.mean())/data.std()
    normalized_data[normalized_data.isna()]=0
    y=["__".join(map(str,el)) for el in data.index] if len(data.index.names)>1 else list(data.index)
    ref_key = ref.index.unique()[0]
    cp_keys = list(control_p.index.unique())
    trial_keys = list(trial.index.unique())
    norm_trial = normalized_data.loc[trial_keys]
    lda_cpdict = {}
    for el in cp_keys:
        lda = sklearn_LDA(n_components=1)
        sub_data = normalized_data.loc[[el,ref_key]]
        sub_y = y=["__".join(map(str,y_)) for y_ in sub_data.index] if len(sub_data.index.names)>1 else list(sub_data.index)
        lda.fit(sub_data,sub_y)
        lda_cpdict[el]=norm_trial.groupby(norm_trial.index.names).apply(LDAcpScale,lda,el)
        
    return lda_cpdict
    
def LDAcpScale(trial,lda,cp_key):
    """ train a LDA on a negativ control and a positiv control and predict a trials outcome

    Args:
        trial (pandas.Dataframe): each row is a data point, each column a descriptor of each trial condition
        ref (pandas.Dataframe): each row is a data point, each column a descriptor of ref condition
        control_p (pandas.Dataframe): each row is a data point, each column a descriptor of a positiv control condition

    Returns:
        array : positive control likelihood for each trial for a control
    """
    cp_key = "__".join(map(str,cp_key)) if type(cp_key) is not str else cp_key
    index = list(lda.classes_).index(cp_key)
    return np.mean(lda.predict_proba(trial)[:,index])

def compareDensityPlot(
    x,y,indexGroupKeys=None,n=50,sigma=2,showDensityMap = False, cmap='gray',filterExtrema = 0.5,
    proportions = [0.1,0.25,0.5,0.75,0.9], modes = "auto",xLabel=None, yLabel=None
    ):

    if xLabel is None:
        xLabel = x.name if hasattr(x,"name") else "Feature1"
    if yLabel is None:
        yLabel = y.name if hasattr(y,"name") else "Feature2"
    X = y;Y=x
    YMask = ExtremaMask(Y,filterExtrema)
    XMask= ExtremaMask(X,filterExtrema)
    YMasked = Y[YMask&XMask]
    XMasked = X[YMask&XMask]
    extent = [YMasked.min(),YMasked.max(),XMasked.min(), XMasked.max(),]
    fig, axes = plt.subplots(2,2,sharex="col", sharey="row",
        gridspec_kw={  'hspace': 0, 'wspace': 0, 'width_ratios': [10, 1], 'height_ratios': [1, 10]}
        )
    axes[0,0].axis("off")
    axes[0,1].axis("off")
    axes[1,1].axis("off")
    colorList = [(1,0,0), (0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,0.5,0.5),(0.5,1,0.5),(0.5,0.5,1)]
    
    widthList = [2,1.5,1,0.8,0.6,0.5,0.4,0.3,0.2]
    styleList = ["-","--","-.",":","-","--","-.",":"]
    alphaList = [1,0.8,0.6,0.4,0.2,0.1,0.05,0.025,0.01]

    if modes == "auto" or (type(modes) is not list and type(modes) is not tuple):
        modes = ["width", "alpha"]

    if indexGroupKeys is None:
        indexGroupKeys = X.index.unique()
    multiple =  len(indexGroupKeys) < len(X) and len(indexGroupKeys)>1

    if not multiple and len(indexGroupKeys) == len(X):
        indexGroupKeys = [indexGroupKeys]
    
    propRange = range(len(proportions))
    if showDensityMap:
        densityMap =  DensityMap(YMasked.loc[indexGroupKeys],XMasked.loc[indexGroupKeys],n,sigma)
        x_mesh,y_mesh = np.linspace(extent[0],extent[1],densityMap.shape[1]),np.linspace(extent[2],extent[3],densityMap.shape[0])
        axes[1,0].pcolor(x_mesh,y_mesh,(densityMap*1.0),cmap=cmap)

    for i, groupKey in enumerate(indexGroupKeys):
        col = colorList[i%len(colorList)]
        densityMap = DensityMap(YMasked.loc[groupKey],XMasked.loc[groupKey],n,sigma)
        sns.kdeplot(y=XMasked.loc[groupKey],clip=(extent[2],extent[3]),ax=axes[1,1], color=col)
        sns.kdeplot(YMasked.loc[groupKey],clip=(extent[0],extent[1]),ax=axes[0,0], color=col)
        pathPatchKwargs={
            "edgecolor":[col+(alphaList[j],)for j in propRange] if "alpha" in modes else col+(1,),
            'ls':[styleList[j] for j in propRange] if "style" in modes else "-",
            'facecolor':col+(0.05,) if "fill" in modes else (0,0,0,0),
            'lw':[widthList[j] for j in propRange] if "width" in modes else 2,
            'zorder':[j+1 for j in propRange[::-1]]
            }
        ContourProp(densityMap,axes[1,0],pathPatchKwargs=pathPatchKwargs,extent=extent, proportions=proportions, addEdgeValues=False)


    axes[1,0].set_xlabel(xLabel)
    axes[1,0].set_ylabel(yLabel)
    axes[1,1].set_ylim((extent[2],extent[3]))
    axes[0,0].set_xlim((extent[0],extent[1]))
    # axes[1,0].set_aspect(1/((extent[3]-extent[2])/(extent[1]-extent[0])))

    deffaultCol = (0,0,0) if multiple else (1,0,0)

    propHandle = [
        plt.Line2D([],[],
            color= deffaultCol+((alphaList[j],) if "alpha" in modes else (1,)),
            linewidth = widthList[j] if "width" in modes else 2,
            linestyle = styleList[j] if "style" in modes else "-",
            label = "%d %%"%np.round(proportions[j]*100))
        for j in propRange]
    groupHandle = [
        plt.Line2D([0],[0],marker='o', markersize=8,
            color='w', 
            markerfacecolor=colorList[i%len(colorList)],
            label = str(indexGroupKeys[i]))
        for i in range(len(indexGroupKeys))] if multiple else []
    

    axes[1,0].legend(handles=groupHandle+propHandle,handlelength = 2,fontsize = "small",labelspacing = 0.2,
            shadow=True)
    return fig, axes

def KdePlot(X,Y):
    fig, axe = plt.subplots(1,1,)
    X = np.array(X)
    Y = np.array(Y)
    xmin = X.min()
    xmax = X.max()
    ymin = Y.min()
    ymax = Y.max()
    X_mesh, Y_mesh = np.mgrid[xmin:xmax:20j, ymin:ymax:20j]
    positions = np.vstack([X_mesh.ravel(), Y_mesh.ravel()])
    values = np.vstack([X, Y])
    kernel = stats.gaussian_kde(values)
    kdeVals = kernel(positions).T
    Z = np.reshape(kdeVals, X_mesh.shape)
    axe.imshow(np.rot90(Z), extent=[xmin, xmax, ymin, ymax])
    axe.set_aspect('equal', 'box')
    return fig, axe

def DensityMap(X,Y,n, sigma = None):
    X = np.array(X)
    Y = np.array(Y)
    xmin = X.min()
    xmax = X.max()
    ymin = Y.min()
    ymax = Y.max()
    X = (n*(X - xmin)/(xmax-xmin)).astype(np.uint8)
    Y = (n*(Y - ymin)/(ymax-ymin)).astype(np.uint8)    
    densityMap = np.zeros((n+1,n+1),dtype = np.uint32)
    XY = np.stack([X,Y])
    points, counts=np.unique(XY, axis=1,return_counts=True)
    for (x,y),count in  zip(points.T,counts):
        densityMap[x,y]=count
    if sigma is not None:
        densityMap = gaussian(densityMap,sigma)
        densityMap *= len(X)/np.sum(densityMap)
    return densityMap.T

def DensityPlot(densityMap,axe,*args,**kwargs):

    axe.imshow(densityMap,*args,**kwargs)#, extent=[xmin, xmax, ymin, ymax])
    
    return axe

    # axe.set_aspect('box')

def ContourProp(densityMap,axe, proportions = [0.5,0.4,0.3,0.2,0.1,0.05,0.01],
    pathPatchKwargs={'facecolor':'none','aa':True},extent=None,addEdgeValues=True):

    if extent is not None:
        h,w = densityMap.shape
        xmin,xmax,ymin,ymax = extent
        axe.set_xlim(xmin,xmax)
        axe.set_ylim(ymin,ymax)
    points = np.stack(densityMap.nonzero())
    counts = densityMap[densityMap.nonzero()]
    proportionsIso = IsoFromProp(counts,proportions)
    pathPatchKwargsDict = refactorKwargs(proportions,pathPatchKwargs,["lw","linewidth","linestyle","ls","edgecolor","facecolor","zorder"],["edgecolor","facecolor"])
    
    if 'lw' not in pathPatchKwargs:
        lw = [i+1 for i in range(proportions)]
    else:
        lw = pathPatchKwargs.pop("lw")

    
    for containsPropTarget in proportions:
        contour = axe.contour(densityMap,colors="white",levels=[proportionsIso[containsPropTarget]],linewidths=0, antialiased=True, corner_mask=False)
        paths = sum([el._paths for el in contour.collections if el._paths !=[]],[])
        localKwargs = pathPatchKwargsDict[containsPropTarget]
        for i, path in enumerate(paths):
            path = path.copy()
            if extent is not None:
                path.vertices = (path.vertices / [h,w]) * [xmax-xmin,ymax-ymin] + [xmin,ymin]
            if i ==0 and "label" not in localKwargs:
                localKwargs["label"] = "%d %%"%np.round(containsPropTarget*100)
            elif "label" in localKwargs:
                localKwargs.pop("label")
            axe.add_patch(patches.PathPatch(path, **localKwargs))
            if addEdgeValues:
                x,y = path.vertices[np.random.randint(len(path.vertices))]
                axe.text(x,y,"%d %%"%np.round(containsPropTarget*100),fontweight='bold',zorder=4,
                color=pathPatchKwargs['edgecolor'] if 'edgecolor' in pathPatchKwargs else 'black')
    return axe

def IsoFromProp(counts,proportions = [0.9,0.5,0.2,0.1]):
    sortedCounts = np.sort(counts)[::-1]
    cumProp= np.cumsum(sortedCounts)/np.sum(counts)
    proportionsIso = {}
    for containsPropTarget in proportions:
        j = np.argmin(abs(cumProp-containsPropTarget))
        proportionsIso[containsPropTarget] = sortedCounts[j]
    return proportionsIso

def refactorKwargs(listKey,kwargs,propKeys,listOrTuplePropKeys=[]):
    keyKwargDict = {key:kwargs.copy() for key in listKey}
    for propKey in propKeys:
        if propKey in kwargs:
            el = kwargs[propKey]
            try:
                assert type(el) == list or type(el)==tuple
                if propKey in listOrTuplePropKeys:
                    assert type(el[0]) == list or type(el[0])==tuple
                if len(el)!=len(listKey) and len(el)>0:
                    for key in listKey:
                        keyKwargDict[key][propKey] = el[0]
                elif len(el)==0:
                    for key in listKey:
                     keyKwargDict[key].drop(propKey)
                for key,val in zip(listKey,el):
                    keyKwargDict[key][propKey] = val
            except AssertionError: # not iterator we kept it in the copy
                pass
    return keyKwargDict

def ClassicalScatter(X,Y):
    fig, axe = plt.subplots(1,1)
    axe.scatter(X,Y)
    return fig, axe

def HexBin(X,Y):
    fig, axe = plt.subplots(1,1)
    axe.hexbin(X,Y)
    return fig, axe

def ExtremaMask(X,q=0.5):
    q = max(min(q,1),0)
    q1,q2,q3 = np.quantile(X.dropna(),[0.5-q/2,0.5,0.5+q/2])
    iqr = q3-q1
    minVal = max(q1 - 1.5*iqr,np.min(X))
    maxVal = min(q3 + 1.5*iqr,np.max(X))
    return (X<=maxVal) & (X>minVal)

def absMax(data):
    "signed values of absolut max (further for 0) for each column of a dataframe"
    ids = np.argmax(abs(data.values),axis=0)
    out = pd.DataFrame([data.values[ids[i],i]for i in range(len(ids))],index = data.columns)
    return out.T

def absMin(data):
    "signed values of absolut min (closer to 0)) for each column of a dataframe"
    ids = np.argmin(abs(data.values),axis=0)
    out = pd.DataFrame([data.values[ids[i],i]for i in range(len(ids))],index = data.columns)
    return out.T

def ScatterPlot(data=None,vectors=None,point=None,  title="PCA Scatter Plot", score=None):
    """Build a 3d scatterplot (or call ScatterPlot2d,BoxPlot1D or VectorPlot1D)

    Args:
        data (pd.Dataframe, optional): each row is a data point, each column is an ax of the scatter plot.
            Will be shown as dots on the scatterplot. Defaults to None.
        vectors (pd.Dataframe, optional): each row is a descriptor, each column its corelation with an ax of the scatter plot. Defaults to None.
            Will be shown as vectors on the scatterplot. Defaults to None.
        point (pd.Dataframe, optional): A mitoradar Well projected into the scater plot transform space. Defaults to None.
        title (str, optional): title of the scatter plot. Defaults to "PCA Scatter Plot".
        score (_type_, optional): float. for LDA, the cross validation score to None.

    Returns:
        matplotlib.Figure: the plot figure
    """
    if (data is not None and len(data.columns.unique())==1):
        title = title.replace("Scatter Plot","Box Plot")
        return BoxPlot1D(data.iloc[:,:1] if data is not None else None,title)[0]
    elif(vectors is not None and len(vectors.columns.unique())==1):
        title = title.replace("Scatter Plot","Vector Correlation Plot")

        return VectorPlot1D(vectors.iloc[:,:1],  title, score)

    elif (data is not None and len(data.columns.unique())==2) or (vectors is not None and len(vectors.columns.unique())<=2):
        return ScatterPlot2D(data.iloc[:,:2] if data is not None else None,vectors.iloc[:,:2] if vectors is not None else None, point.iloc[:,:2] if point is not None else None,  title, score)

    fig = plt.figure(title,figsize=(15,7))

    ax = fig.add_subplot(projection='3d')
    if data is not None:
        data = data.dropna()
        multi_index = len(data.index.levels)>1 if type(data.index) is pd.core.indexes.multi.MultiIndex else False
        # ax2 = fig.add_subplot()
        edge_color_dict = {key:COLORS_STRONG[i%len(COLORS_STRONG)] for i, key in enumerate(data.index.levels[0] if multi_index else data.index.unique())}
        color_dict = {key:COLORS_LIGHT[i%len(COLORS_LIGHT)] for i, key in enumerate(data.index.unique().sort_values())}
        edge_colors = [edge_color_dict[key[0] if multi_index else key] for key in data.index] 
        colors = [color_dict[key] for key in data.index]
    if vectors is not None:
        multi_index2 = len(vectors.index.levels)>1 if type(vectors.index) is pd.core.indexes.multi.MultiIndex else False
        objs = [el.split("_")[0] for el in (vectors.index.levels[0] if multi_index2 else vectors.index)]
        dark_color_dict = {key:COLORS_DARK[i%len(COLORS_DARK)] for i, key in enumerate(np.unique(objs))}
        dark_colors = [dark_color_dict[key[0] if multi_index2 else key] for key in objs]
    
    columns = data.columns if data is not None else vectors.columns
    if data is not None:
        ax.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2], c=colors, edgecolors = edge_colors, linewidths=1.5, alpha=0.9)
    if vectors is not None:
        # ax.scatter(data[columns[0]],data[columns[1]],data[columns[2]], c=colors, edgecolors = edge_colors, linewidths=1.5, alpha=0.9)
        if data is not None:
            scf = [float(data[columns[i]].max()-data[columns[i]].min())/2 for i in range(3)]
            orig = [float(data[columns[i]].max()+data[columns[i]].min())/2 for i in range(3)] 
            vect_end = pd.DataFrame([(vectors[columns[i]].values*scf[i])+orig[i] for i in range(3)]).T
            orig = pd.DataFrame([orig for j in range(len(vectors))])
        else: 
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            scf = [1,1,1]
            orig = pd.DataFrame(np.zeros((len(vectors),3)))
            vect_end = pd.DataFrame([vectors[columns[i]].values for i in range(3)]).T
        vect_end.columns=columns
        

        hex_col = ['#%02x%02x%02x'%tuple([int(c*255) for c in el]) for el in dark_colors]
        ax.quiver(orig.iloc[:,0],orig.iloc[:,1],orig.iloc[:,2],vect_end.iloc[:,0]-orig.iloc[:,0],vect_end.iloc[:,1]-orig.iloc[:,1],vect_end.iloc[:,2]-orig.iloc[:,2],
            color=hex_col+list(np.repeat(hex_col,2)), arrow_length_ratio=0.1,lw=0.5)


        for i , el in enumerate(vectors.index):

            x,y,z = vect_end.iloc[i,:]
            a,b,c = vectors.iloc[i,:]
            fact = 1.05
            x,y,z = fact*x,fact*y,fact*z
            t = ax.text(x,y,z,el,(a,b,c),fontsize=5,color =hex_col[i],multialignment = "center",backgroundcolor = (1,1,1,0.2))
            if a<0:
                t.set_horizontalalignment("right")
            if b<0:
                t.set_verticalalignment("top")

            # print(x,y,z)
    
    posit = ax.get_position()
    if data is None:
        ax.set_position([0.6*posit.width,0*posit.height,posit.width*1.4,posit.height*1.4])
    else:
        ax.set_position([0.2*posit.width,0*posit.height,posit.width*1.6,posit.height*1.4 ])
        fig.legend(
            handles=[ plt.Line2D(
                [0], [0], marker='o',lw=0,
                color=color_dict[key],
                label=" ".join(map(str,key)) if type(key)==tuple else key,
                markeredgecolor = edge_color_dict[key[0] if multi_index else key],
                markeredgewidth=2,
            ) for key in color_dict],
            ncol = 2 if len(color_dict)>30 else 1,
            fontsize = "small" if len(color_dict)<100 else "x-small",
            labelspacing = 0.2,
            bbox_to_anchor=[0.7,1],
            bbox_transform=fig.transFigure,
            loc='upper left',
            )
    small_label = len(columns[0].split("\n"))>5
    ax.set_xlabel(columns[0],labelpad=20, fontweight="semibold",color="dimgray",fontsize = 'small'if small_label else'medium')
    ax.set_ylabel(columns[1],labelpad=20, fontweight="semibold",color="dimgray")
    ax.set_zlabel(columns[2],labelpad=20, fontweight="semibold",color="dimgray")
    for el in ax.get_xticklabels()+ax.get_yticklabels()+ax.get_zticklabels():
        el.set(alpha=0.5,color="steelblue")
    if data is not None and point is not None:
        ax.plot(*point.values[0],"ro",zorder=40., color="black")
        ax.text(*point.values[0]," "+point.index[0],zorder=40., fontweight="bold",va="center")
    plt.suptitle("%s %s"%(title,("\n 2kCV Accuracy = %d%%"%int(score*100)) if score is not None else ""),x=0.02,y=0.9,fontsize= "large",fontweight= "bold",ha="left")
    return fig

def ScatterPlot2D(data=None,vectors=None,point=None,  title="PCA Scatter Plot", score=None):
    """Build a 2d scatterplot 

    Args:
        data (pd.Dataframe, optional): each row is a data point, each column is an ax of the scatter plot.
            Will be shown as dots on the scatterplot. Defaults to None.
        vectors (pd.Dataframe, optional): each row is a descriptor, each column its corelation with an ax of the scatter plot. Defaults to None.
            Will be shown as vectors on the scatterplot. Defaults to None.
        point (pd.Dataframe, optional): A mitoradar Well projected into the scater plot transform space. Defaults to None.
        title (str, optional): title of the scatter plot. Defaults to "PCA Scatter Plot".
        score (_type_, optional): float. for LDA, the cross validation score to None.

    Returns:
        matplotlib.Figure: the plot figure
    """
    fig = plt.figure(title,figsize=(15,7))

    ax = fig.add_subplot()
    ax.set_box_aspect(1)
    if data is not None:
        data = data.dropna()
        multi_index = len(data.index.levels)>1 if type(data.index) is pd.core.indexes.multi.MultiIndex else False
        # ax2 = fig.add_subplot()
        edge_color_dict = {key:COLORS_STRONG[i%len(COLORS_STRONG)] for i, key in enumerate(data.index.levels[0] if multi_index else data.index.unique())}
        color_dict = {key:COLORS_LIGHT[i%len(COLORS_LIGHT)] for i, key in enumerate(data.index.unique().sort_values())}
        edge_colors = [edge_color_dict[key[0] if multi_index else key] for key in data.index] 
        colors = [color_dict[key] for key in data.index]
    if vectors is not None:
        multi_index2 = len(vectors.index.levels)>1 if type(vectors.index) is pd.core.indexes.multi.MultiIndex else False
        objs = [el.split("_")[0] for el in (vectors.index.levels[0] if multi_index2 else vectors.index)]
        dark_color_dict = {key:COLORS_DARK[i%len(COLORS_DARK)] for i, key in enumerate(np.unique(objs))}
        dark_colors = [dark_color_dict[key[0] if multi_index2 else key] for key in objs]
    
    columns = data.columns if data is not None else vectors.columns
    if data is not None:
        ax.scatter(data.iloc[:,0],data.iloc[:,1],s=None if len(data)<1000 else (1000/len(data))**0.5, c=colors, edgecolors = edge_colors, linewidths=1.5, alpha=0.9)
    if vectors is not None:
        # ax.scatter(data[columns[0]],data[columns[1]],data[columns[2]], c=colors, edgecolors = edge_colors, linewidths=1.5, alpha=0.9)
        if data is not None:
            scf = [float(data[columns[i]].max()-data[columns[i]].min())/2 for i in range(2)]
            orig = [float(data[columns[i]].max()+data[columns[i]].min())/2 for i in range(2)] 
            vect_end = pd.DataFrame([(vectors[columns[i]].values*scf[i])+orig[i] for i in range(2)]).T
            orig = pd.DataFrame([orig for j in range(len(vectors))])
        else: 
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            scf = [1,1]
            orig = pd.DataFrame(np.zeros((len(vectors),2)))
            vect_end = pd.DataFrame([vectors[columns[i]].values for i in range(2)]).T
        vect_end.columns=columns
        

        hex_col = ['#%02x%02x%02x'%tuple([int(c*255) for c in el]) for el in dark_colors]
        if data is not None:
            ax.quiver(orig.iloc[:,0],orig.iloc[:,1],vect_end.iloc[:,0]-orig.iloc[:,0],vect_end.iloc[:,1]-orig.iloc[:,1],
                color=hex_col,lw=0.5,scale_units="xy",scale = 1,width = 0.003)# arrow_length_ratio=0.1,lw=0.5)
        else:
            ax.quiver(orig.iloc[:,0],orig.iloc[:,1],vect_end.iloc[:,0]-orig.iloc[:,0],vect_end.iloc[:,1]-orig.iloc[:,1],
                color=hex_col,lw=0.5,scale = 2,width = 0.003)# arrow_length_ratio=0.1,lw=0.5)


        for i , el in enumerate(vectors.index):

            x,y = vect_end.iloc[i,:]
            a,b = vectors.iloc[i,:]
            fact = 1.05
            x,y = fact*x,fact*y
            t = ax.text(x,y,el,fontsize=8,color =hex_col[i],ha = "center",va="center",backgroundcolor = (1,1,1,0.2))
            if b>0:
                t.set_verticalalignment("bottom")
            if b<0:
                t.set_verticalalignment("top")

            # print(x,y,z)
    
    posit = ax.get_position()
    if data is None:
        pass
    else:
        fig.legend(
            handles=[ plt.Line2D(
                [0], [0], marker='o',lw=0,
                color=color_dict[key],
                label=" ".join(str(key)) if type(key)==tuple else key,
                markeredgecolor = edge_color_dict[key[0] if multi_index else key],
                markeredgewidth=2,
            ) for key in color_dict],
            ncol = 2 if len(color_dict)>30 else 1,
            fontsize = "small" if len(color_dict)<100 else "x-small",
            labelspacing = 0.2,
            bbox_to_anchor=[0.7,1],
            bbox_transform=fig.transFigure,
            loc='upper left',
            )
    small_label = len(columns[0].split("\n"))>5
    ax.set_xlabel(columns[0],labelpad=20, fontweight="semibold",color="dimgray",fontsize = 'small'if small_label else'medium')
    ax.set_ylabel(columns[1],labelpad=20, fontweight="semibold",color="dimgray")
    for el in ax.get_xticklabels()+ax.get_yticklabels():
        el.set(alpha=0.5,color="steelblue")
    if data is not None and point is not None:
        ax.plot(*point.values[0],"ro",zorder=40., color="black")
        ax.text(*point.values[0],"  "+point.index[0],zorder=40., fontweight="bold")
    plt.suptitle("%s %s"%(title,("\n 2kCV Accuracy = %d%%"%int(score*100)) if score is not None else ""),x=0.1,y=0.9,fontsize= "large",fontweight= "bold",ha="left")
    plt.tight_layout()
    return fig

def BoxPlot1D(data,title="PCA BoxPlot",color="Custom"):
    """Build a 1d boxplot with one box for each condition 

    Args:
        data (pd.Dataframe, optional): each row is a data point, one column wich is an ax (LDA, PCA) or a descriptor.
            Will be shown as dots on the scatterplot. Defaults to None.
        vectors (pd.Dataframe, optional): Useless, kept for consistency. Defaults to None.
        point (pd.Dataframe, optional): Useless, kept for consistency. Defaults to None.
        title (str, optional): title of the scatter plot. Defaults to "PCA Scatter Plot".
        score (_type_, optional): Useless, kept for consistency. Defaults to None.

    Returns:
        matplotlib.Figure: the plot figure
    """
    fig = plt.figure(title,figsize=(15,7))
    ax = fig.add_subplot()
    ax.set_title(title)

    data = data.dropna()
    data.sort_index(inplace=True)
    multi_index = len(data.index.levels)>1 if type(data.index) is pd.core.indexes.multi.MultiIndex else False
    keys = data.index.unique()
    if color == "Custom":
        edge_color_dict = {key:COLORS_STRONG[i%len(COLORS_STRONG)] for i, key in enumerate(data.index.levels[0] if multi_index else data.index.unique())}
        color_dict = {key:COLORS_LIGHT[i%len(COLORS_LIGHT)] for i, key in enumerate(keys.sort_values())}
    elif color == "Default":
        color_dict = {key:"C%d"%i for i, key in enumerate(keys.sort_values())}
        edge_color_dict = {key:"C%d"%i for i, key in enumerate(data.index.levels[0] if multi_index else data.index.unique())}
    else:
        color_dict = {key:color for i, key in enumerate(keys.sort_values())}
        edge_color_dict = {key:color for i, key in enumerate(data.index.levels[0] if multi_index else data.index.unique())}
    edge_colors = [edge_color_dict[key[0] if multi_index else key] for key in data.index] 
    colors = [color_dict[key] for key in data.index]

    bplot = ax.boxplot([data.loc[key].values.flatten() for key in keys],
        vert=True,  # vertical box alignment
        patch_artist=True,  # fill with color
    )
    for med, wisk1,wisk2, caps1, caps2, patch, el in zip(bplot['medians'],bplot['whiskers'][::2],bplot['whiskers'][1::2],bplot['caps'][::2],bplot['caps'][1::2],bplot['boxes'], keys):
        patch.set_facecolor(color_dict[el])
        patch.set_edgecolor(edge_color_dict[el[0] if multi_index else el])
        wisk1.set_color("black")
        caps1.set_color("black")
        wisk2.set_color("black")
        caps2.set_color("black")
        med.set_color("black")
    for flier in bplot["fliers"]:
        flier.set_marker(".")
        flier.set_markerfacecolor("black")
        flier.set_markeredgecolor("black")

    ax.set_xticks(np.arange(1, len(keys) + 1), labels=[str(k).translate({ord(el):""for el in ["(",")",",","'",'"']}) for k in keys])
    ax.set_ylabel(data.columns[0].translate({ord(el):""for el in ["(",")",",","'",'"']}),labelpad=20, fontweight="semibold")
    plt.setp(ax.get_xticklabels(), rotation=10, horizontalalignment='right')
    out = data.groupby( data.index).agg(["mean","median","std","min","max"])
    return fig,out

def ViolinPlot(data=None,title="ViolinPlot",color="Custom"):
    fig = plt.figure(title,figsize=(15,7))
    ax = fig.add_subplot()
    ax.set_title(title)
    data = data.dropna().sort_index()

    keys = data.index.unique()


    if color == "Custom":
        color_dict = {key:COLORS_LIGHT[i%len(COLORS_LIGHT)] for i, key in enumerate(keys.sort_values())}
    elif color == "Default":
        color_dict = {key:"C%d"%i for i, key in enumerate(keys.sort_values())}
    else:
        color_dict = {key:color for i, key in enumerate(keys.sort_values())}

    data = data.dropna().sort_index()
    multi_index = len(data.index.levels)>1 if type(data.index) is pd.core.indexes.multi.MultiIndex else False

    vplot = ax.violinplot([data.loc[key].values.flatten() for key in keys],
        vert=True, showextrema=False,  # vertical box alignment
    )
    for b,k in zip(vplot["bodies"], keys):
        b.set_color(color_dict[k])
        if color == "Default": b.set_color(b._facecolors[0][:3]) #strange but needed
    out = data.groupby( data.index).agg(["mean","median","std","min","max"])
    ax.set_xticks(np.arange(1, len(keys) + 1), labels=[str(k).translate({ord(el):""for el in ["(",")",",","'",'"']}) for k in keys])
    ax.set_ylabel(data.columns[0].translate({ord(el):""for el in ["(",")",",","'",'"']}),labelpad=20, fontweight="semibold")
    plt.setp(ax.get_xticklabels(), rotation=10, horizontalalignment='right')

    return fig, out

def BarPlot(data=None,title="BarPlot",color="Custom"):
    fig = plt.figure(title,figsize=(15,7))
    ax = fig.add_subplot()
    ax.set_title(title)
    data = data.dropna().sort_index()

    keys = data.index.unique()
    
    if color == "Custom":
        color_dict = {key:COLORS_LIGHT[i%len(COLORS_LIGHT)] for i, key in enumerate(keys.sort_values())}
    elif color == "Default":
        color_dict = {key:"C%d"%i for i, key in enumerate(keys.sort_values())}
    else:
        color_dict = {key:color for i, key in enumerate(keys.sort_values())}

    data = data.dropna()
    data.sort_index(inplace=True)
    multi_index = len(data.index.levels)>1 if type(data.index) is pd.core.indexes.multi.MultiIndex else False
    out = data.groupby( data.index).agg(["mean","std"])
    out.columns = out.columns.droplevel(0)
    out["ste"] = out["std"]/(len(out)**0.5)
    yerr = out.loc[keys,"ste"].values
    yerr = [[el/10 for el in yerr] , [el for el in yerr]]
    X = [i+1 for i in range(len(keys))]
    Y = [data.loc[key].values.mean() for key in keys]
    ax.errorbar(X,Y, yerr=yerr, capsize=10,fmt='none', color="black")

    vplot = ax.bar(X,Y, color=[color_dict[key] for key in keys],zorder=2.5
                    # yerr = yerr, capsize=10
                   )
    
    # ax.get_xaxis().set_visible(False)
    ax.set_xticks(np.arange(1, len(keys) + 1), labels=[str(k).translate({ord(el):""for el in ["(",")",",","'",'"']}) for k in keys])
    ax.set_ylabel(data.columns[0].translate({ord(el):""for el in ["(",")",",","'",'"']}),labelpad=20, fontweight="semibold")
    plt.setp(ax.get_xticklabels(), rotation=10, horizontalalignment='right')
    fig.tight_layout()
    return fig, out

def VectorPlot1D(vectors=None, title="PCA Vector Plot", score=None):
    """Build a 1d boxplot with one arrow for each descriptor 

    Args:

        vectors (pd.Dataframe, optional): each row is a descriptor, one column wich is it corelation with a PCA or LDA axe. Defaults to None.
            Will be shown as vectors on the scatterplot. Defaults to None.
        title (str, optional): title of the scatter plot. Defaults to "PCA Scatter Plot".
        score (_type_, optional):LDA score. Defaults to None.

    Returns:
        matplotlib.Figure: the plot figure
    """
    fig = plt.figure(title,figsize=(15,7))
    vectors = vectors.sort_values(by=vectors.columns[0],ascending=False)
    vectors["Descriptors_id"] = list(range(1,len(vectors)+1))

    ax = fig.add_subplot()

    multi_index2 = len(vectors.index.levels)>1 if type(vectors.index) is pd.core.indexes.multi.MultiIndex else False
    objs = [el.split("_")[0] for el in (vectors.index.levels[0] if multi_index2 else vectors.index)]
    dark_color_dict = {key:COLORS_DARK[i%len(COLORS_DARK)] for i, key in enumerate(np.unique(objs))}
    dark_colors = [dark_color_dict[key[0] if multi_index2 else key] for key in objs]

    columns = vectors.columns


        # ax.scatter(data[columns[0]],data[columns[1]],data[columns[2]], c=colors, edgecolors = edge_colors, linewidths=1.5, alpha=0.9)

    ax.set_xlim(0, len(vectors)+1)
    ax.set_ylim(-1, 1)
    scf = [1,1]
    
    vect_end = pd.DataFrame([vectors[columns[i]].values for i in range(2)]).T
    orig = vect_end.copy()
    orig.values[:,0]=0
    vect_end.columns=columns
    

    hex_col = ['#%02x%02x%02x'%tuple([int(c*255) for c in el]) for el in dark_colors]

    ax.quiver(orig.iloc[:,1],orig.iloc[:,0],vect_end.iloc[:,1]-orig.iloc[:,1],vect_end.iloc[:,0]-orig.iloc[:,0],
            color=hex_col,angles='xy', scale_units='xy', scale=1,width = 0.003)# arrow_length_ratio=0.1,lw=0.5)


    for i , el in enumerate(vectors.index):

        x= vectors.iloc[i,1]
        s = np.sign(vectors.iloc[i,0])
        y = -0.05*s

        t = ax.text(x,y,el,fontsize=8,color =hex_col[i],va="bottom" if s<0 else "top",backgroundcolor = (1,1,1,0.2), rotation = -90*s, ha = "center")

            # print(x,y,z)
    

    ax.get_xaxis().set_visible(False)
    ax.set_ylabel(columns[0],labelpad=20, fontweight="semibold",color="dimgray")
    for el in ax.get_yticklabels():
        el.set(alpha=0.5,color="steelblue")

    plt.suptitle("%s %s"%(title,("\n 2kCV Accuracy = %d%%"%int(score*100)) if score is not None else ""),x=0.1,y=0.9,fontsize= "large",fontweight= "bold",ha="left")
    
    leg = plt.legend([plt.Line2D([],[]) for i in range(len(dark_color_dict) - 1*("" in dark_color_dict))],
        [obj.capitalize() for obj in (dark_color_dict) if obj !=""],
        handlelength = 0, ncol = len(dark_color_dict), bbox_to_anchor=(0.5, 1),
        loc = 'lower center', title="Objects", title_fontproperties = {"weight":"bold", "size":"large"})
    for text in leg.get_texts():
        text.set_color(dark_color_dict[text._text.lower()])    
    
    plt.tight_layout()
    return fig    

def pvalThresh(data, thresh = [0.05,0.01,0.001], axis=0):
    """generate a *, **, *** percentil representation according to thresholds

    Args:
        data (pandas dataframe): 
        thresh (list, optional): . Defaults to [0.05,0.01,0.001].

    Returns:
        pandas.dataframe: 
    """
    if axis == 1:
        data = data.T
    n = len(data)
    pval_thresh = [100*sum(data<t)/n for t in thresh]

    return {"%s"%('*'*(i+1)):pval_thresh[i] for i in range(len(thresh))}

def buildIndex(data, index_names="all"):
    """transform imformation column into multiindex

    Args:
        data (pandas.DataFrame): raw DataFrame
        index_names (str or list, optional): list of index names, string can be used as shortcut 
            for pre registered list:
                    - "position":["plate", "row", "col", "field"]
                    - "condition":["cell_line", "drug", "dose"]
                    - "all": ["cell_line", "drug", "dose","plate", "row", "col", "field"]
        Defaults to "all".

    Returns:
        pandas.DataFrame: indexed dataframe
    """
    if type(index_names)==str:
        if index_names == "position":
            index_names = ["plate", "row", "col", "field"]
        elif index_names == "condition":
            index_names = ["cell_line", "drug", "dose"]
        else:
            index_names = ["cell_line", "drug", "dose","plate", "row", "col", "field", ]
    return data.set_index(index_names).sort_index()

def removeCol(data, names = ['imID', 'condition', 'date', 'code', 'position']):
    """shortcut to drop columns

    Args:
        data (pandas.DataFrame): raw dataframe or indexed dataframe
        names (list, optional): list of column to drop, if None, all non numercial columns ar droped.
            Defaults to ['imID', 'condition', 'date', 'code', 'position'].

    Returns:
        pandas.DataFrame: dataframe
    """
    if names is None:
        return data.select_dtypes(['number'])
    return data.drop(names, axis=1)

def wellData(data):
    """index and agregate dataframe so that each row corespond to one unique well

    Args:
        data (pandas.DataFrame): raw dataframe

    Returns:
        pandas.DataFrame: indexed, well agregated dataframe
    """
    return removeCol(buildIndex(data)).sort_index().groupby(level = [0,1,2,3,4,5]).mean()

def summaryView(summary, title="Fig 1"):
    """plot a summaryview, heatmap of pvalues os SSMD

    Args:
        summary (pandas.DataFrame): a single dataframe of ssmd or pval with dimention C*D,
            where C is the number of condition and D the number of descriptor. in case of pval, 
            results should be passed throug log10 for beter visualisation
    """
    plt.figure(title, 
    figsize=(14,7)
    )
    is_ssmd = "SSMD" in title
    aspect = min(1, 0.4*summary.shape[1]/summary.shape[0])
    # plt.title(title)

    plt.rc('xtick', labelsize=6) 
    plt.rc('ytick', labelsize=8)
    if is_ssmd:
        current_cmap = matplotlib.colormaps["seismic"].copy()
        current_cmap.set_bad(color='lightgray')
        im = plt.imshow(summary, cmap=current_cmap, vmin=-7 , vmax=7 , aspect = aspect)
    else:
        current_cmap = matplotlib.colormaps["YlGn"].copy().reversed()
        current_cmap.set_extremes(bad='lightgray', under=current_cmap(0), over=current_cmap(0.9999))
        if summary.isna().values.all():
            max_ = 1
            min_ = 10**-4
        else:
            max_ = min(10**(np.ceil(np.log10(np.nanmax(summary.values)))),1)
            min_ = np.nanmin(summary.values)
            if min_>0: min_ = 10**(np.floor(np.log10(min_)))
            min_ = max_*10**-4 if min_>max_*10**-4 else min_ if min_>max_*10**-10 else max_*10**-10
        im= plt.imshow(summary, norm = matplotlib.colors.LogNorm(min_,max_,True),cmap=current_cmap, aspect = aspect)
    cbar = plt.colorbar(im, shrink = 0.3)
    im_height = im.get_window_extent().bounds[-1]

    s_desc = summary.columns
    objs = [desc.split("_")[0] if "_" in desc else "" for desc in s_desc]
    objs_col = {obj:COLORS_DARK[i%len(COLORS_DARK)] for i, obj in enumerate(np.unique((objs)))}
    cols = [objs_col[obj] for obj in objs]
    locs, labels = plt.xticks(range(len(summary.columns)),summary.columns, rotation=90)
    for i in range(len(labels)):
        labels[i].set_color(cols[i])
    s_index =[" ".join(map(str,el)) if type(el) is tuple else str(el) for el in summary.index]
    if summary.shape[0]>50:
        plt.yticks(range(len(s_index)),s_index, fontsize=int(650/summary.shape[0]))
    else:
        plt.yticks(range(len(s_index)),s_index)
    leg = plt.legend([plt.Line2D([],[]) for i in range(len(objs_col) - 1*("" in objs_col))],
        [obj.capitalize() for obj in (objs_col) if obj !=""],
        handlelength = 0, ncol = len(objs_col), bbox_to_anchor=(0.5, 1),
        loc = 'lower center', title=title, title_fontproperties = {"weight":"bold", "size":"large"})
    leg_obj_item = [el for el in np.unique((objs)) if el!=""]
    for i, text in enumerate(leg.get_texts()):
        text.set_color(objs_col[leg_obj_item[i]])
    
    plt.tight_layout()

def spiderplot(data,sinif = None,  ax=None, fig_kw = {"num":1}, lim = (-6,6), thresh = [-3,-2,-1,0,1,2,3], control_name = "Control", shiny=True, sort_val = True):
    """Generate a simple Radar Plot

    Args:
        data (pandas.Dataframe): SSMD data to be added on the spiderplot each row is a condition, each column a descriptor
        sinif (pandas.Dataframe): u_values corresponding to data each row is a condition, each column a descriptor
        ax (matplotlib.axes.Axes, optional): ax to plot the datas. Defaults to None.
        fig_kw (dict, optional): optional keword for the figure object. Defaults to {"num":1}.
        lim (tuple): SSMD lim of the radar plot
        thresh: deprecated
        control_name (str): control name
        shiny (bool) should be default for nice looking radar plot
    """
    build_fig = ax is None
    data["Sort_col"] = absMax(data.T).values[0]
    if sinif is not None:
        sinif["Sort_col"] = data["Sort_col"]
        sinif = sinif.sort_values("Sort_col") if sort_val else sinif.sort_index()
        sinif = sinif.drop("Sort_col",axis=1)
    data=data.sort_values("Sort_col") if sort_val else data.sort_index()
    data=data.drop("Sort_col",axis=1)
    #values = data.to_list()
    labels = data.index.to_list()
    title = labels[0].split("_")[0]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # values += values[:1]
    angles += angles[:1]
    #values = np.array(values)
    # values[values>lim[1]]=lim[1]
    # values[values<lim[0]]=lim[0]
    data.values[data.values>lim[1]]=lim[1]
    data.values[data.values<lim[0]]=lim[0]

    if build_fig:
        fig, ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(polar=True), **fig_kw, )
    else:
        fig=None
    ax.set_ylim(lim[0]-1.8*lim[1]/6, lim[1]+1.8*lim[1]/6)
    add_circle(ax,shiny=shiny)
    ax.scatter(angles, [0 for _ in angles], color="blue", s=4)
    for i, col in enumerate(data.columns):
        values = data[col].to_list()
        values += values[:1]
        if sinif is not None:
            sinif_values = sinif[col].to_list()
            sinif_values += sinif_values[:1]
        else:
            sinif_values = values*0 + 5

        ax.plot(angles, values, color=COLORS_STRONG[::-1][(i+1)%len(COLORS_STRONG)], linewidth=0.75)
        ax.scatter(angles, values, color=COLORS_STRONG[::-1][(i+1)%len(COLORS_STRONG)],edgecolors="black", linewidths=0.3, s=sinif_values)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    names = [lab.replace("%s_"%title,"") for lab in labels]
    ids = [i+1 for i in range(len(names))]

    if build_fig:
        ax.set_thetagrids(np.degrees(angles[:-1]), names)
    else:
        ax.set_thetagrids(np.degrees(angles[:-1]), ids, fontsize = 5)
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
        if not build_fig:
            label._y = 0.08
    for label in ax.get_yticklabels():
        if not build_fig:
            label.set_fontsize(5)
    if build_fig:
        botom = fig.supxlabel('Legend Placeholder')
        botom.set_alpha(0)
        handles = generateHandle(data,control_name,shiny=shiny)
        fig.legend(
            handles=handles,handlelength = 2,fontsize = "small",labelspacing = 0.2,shadow=True,
            ncol = 5,loc='lower right',mode = "expand" )
        fig.set_tight_layout(True)
        return fig

    else:
        legend = ["%d: %s"%(i, name) for i, name in zip(ids,names)]
        lg = ax.legend(handles=[plt.Line2D([],[],linewidth=0) for _ in range(len(names))],
            labels = legend,handlelength = 0,title = title.capitalize(),
            title_fontsize = "small",loc='center left',  bbox_to_anchor=(1.02, 0.5),
            fontsize = "xx-small",labelspacing = 0.2,frameon=False,)
        if shiny:
            c_map = matplotlib.colormaps["coolwarm"]
            v_max = max(abs(lim[0]),abs(lim[1]))
            for l,t, v in zip(ax.get_xticklabels(),lg.get_texts(),absMax(data.T).values[0]):
                t.set_color(c_map(v/v_max+0.5))
                t.set_path_effects(path_effects=[path_effects.withSimplePatchShadow((0.3, -0.3),"black",alpha=1, rho=0.5)])
                l.set_color(c_map(v/v_max+0.5))
                l.set_path_effects(path_effects=[path_effects.withSimplePatchShadow((0.3, -0.3),"black",alpha=1, rho=0.5)])
        return ax

def add_circle(ax, ref_val = 0 , thresh_pos=[1,2,3], thresh_neg=[-1,-2,-3], shiny=True):
    """add background circle to a radar plot according to thresholds

    Args:
        ax (matplotlib.Ax): _description_
        ref_val (int, optional): control value. Defaults to 0.
        thresh_pos (list, optional): positiv threshold values. Defaults to [1,2,3].
        thresh_neg (list, optional):  negativ threshold values. Defaults to [-1,-2,-3].
    """
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False).tolist()
    angles += angles[:1]
    val = np.array(angles)*0 + 1

    ylim=ax.get_ylim()
    if shiny:
        ax.grid(False)
        radii = np.linspace(*ylim,64)
        thetas_radians = np.arange(0,2.01*np.pi,np.pi/100.)
        grad = np.ones((64,))*0.5
        grad[radii<thresh_neg[0]] = np.linspace(0.,0.45,sum(radii<thresh_neg[0]))
        grad[radii>thresh_pos[0]] = np.linspace(0.55,1.,sum(radii>thresh_pos[0]))
        grad = (np.atleast_2d(grad).T)
        grad = grad.repeat(len(thetas_radians), axis=1)
        ax.pcolormesh(thetas_radians,radii,grad,edgecolors='face',cmap="coolwarm",shading='gouraud')
        ax.grid(True,color='white',alpha=0.2)
        ax.plot(angles, val*thresh_pos[0], color='gray', alpha=0.2,linewidth=0.3)
        ax.plot(angles, val*thresh_neg[0], color='gray', alpha=0.2,linewidth=0.3)
    else:
        ax.fill(angles, val*thresh_pos[0], color='gray', alpha=0.3)
        ax.fill(angles, val*thresh_neg[0], color='white', alpha=1)
        ax.plot(angles, val*ref_val, color='blue', linewidth=1)

        step = 1/len(thresh_pos)
        for i, el in enumerate(thresh_pos):
            ax.plot(angles, val*el, color='darkred', linewidth=1, linestyle="--", alpha = step*(i+0.5))
        step = 1/len(thresh_neg)
        for i, el in enumerate(thresh_neg):
            ax.plot(angles, val*el, color='darkblue', linewidth=1, linestyle="--", alpha = step*(i+0.5))

    ax.plot(angles, val*ref_val, color='blue', linewidth=0.75)

def spiderplotMulty(data,sinif=None, control_name = "Control", one_fig = True, shiny=True, lim = (-6,6), thresh = 0.05, sort_val = True):
    """produce muliple spiderplot, one for each target (cell - mito ...)

    Args:
        data (_type_): _description_
        control_name (str, optional): name of the control. Defaults to "Control".
        one_fig (bool, optional): if true, everithing is ploted on one fig, else for each target a new fig is ploted. Defaults to True.
    Returns:
        list of fig
    """
    data = pd.DataFrame(data)
    sinif = pd.DataFrame(sinif)
    index = data.index.to_list()
    data["obj"] = [el.split("_")[0] for el in index]
    obj = [el for el in data["obj"].unique()]
    out = []
    if len(obj)>0: 
        obj=["Measures"]
        data["obj"] = "Measures"

    if one_fig:
        subplot_shape = (1+(len(obj)>3),3 if (len(obj)>2) else len(obj))

        fig, axes = plt.subplots(*subplot_shape, figsize=(15,7), subplot_kw=dict(polar=True), constrained_layout=True)
        if subplot_shape[1]==1:
            axes = np.array([axes])
        for i in range(len(axes.flatten())):
            if i>=len(obj):
                fig.delaxes(axes.flatten()[i])
        out.append(fig)
    
    for i, el in enumerate(obj):
        to_plot = data[data["obj"]==el]
        sinif_to_plot = sinif[data["obj"]==el] if sinif is not None else None
        to_plot = to_plot[[col for col in data.columns if "obj" not in col]]
        sinif_to_plot = sinif_to_plot[[col for col in data.columns if "obj" not in col]]
        if one_fig:
            spiderplot(to_plot,sinif_to_plot, fig_kw={"num":el}, ax=axes.flatten()[i], shiny=shiny,lim=lim, sort_val=sort_val)
        else:
            fig = spiderplot(to_plot,sinif_to_plot, fig_kw={"num":el}, control_name=control_name,shiny=shiny,lim=lim,sort_val=sort_val)
            out.append(fig)
    if one_fig:
        fig.canvas.manager.set_window_title('MitoRadar SSMD Overview') 
        botom = fig.supxlabel('Legend Placeholder')
        botom.set_alpha(0)
        handles = generateHandle(data,control_name,shiny=shiny,thresh=thresh)
        fig.legend(handles=handles,handlelength = 2,fontsize = "small",labelspacing = 0.2,
            shadow=True,ncol = min(len(handles),8),loc='lower right',mode = "expand")
    return out
        
def generateHandle(data,control_name,shiny=True,thresh=0.05):
    """generate handle legend for a radar plot

    Args:
        data (pandas.Dataframe): SSMD data to be added on the spiderplot each row is a condition, each column a descriptor
        control_name (str): control name
        shiny (bool) for nice looking radar plot
        thresh (float, optional): significativity threshold. Defaults to 0.05.

    Returns:
        list(matplotlib.pyplot.Line2D): legend handels
    """
    if shiny:
        return [
            plt.Line2D([],[],color= COLORS_STRONG[::-1][(i+1)%len(COLORS_STRONG)],
            label = smartIndex([col])[0],)
            for i, col in enumerate(data.columns) if "obj" not in col]+[
            plt.Line2D([],[],color= "blue", label = control_name),
            plt.Line2D([],[],color= "blue", linewidth=4, alpha = 0.2, label = "Diminishing effect"),
            plt.Line2D([],[],color= "red", linewidth=4, alpha = 0.2, label = "Increasing effect"),
            plt.Line2D([],[],color= "gray", linewidth=4, alpha = 0.2, label = "Low effect"),
            plt.Line2D([0], [0], marker='o', color='w', label='p>%s .'%smartFloat2Str(thresh,5),markerfacecolor='w',markeredgecolor="black", markersize=2.5),
            plt.Line2D([0], [0], marker='o', color='w', label='p<%s *'%smartFloat2Str(thresh,5),markerfacecolor='w',markeredgecolor="black", markersize=3),
            plt.Line2D([0], [0], marker='o', color='w', label='p<%s **'%smartFloat2Str(thresh/5,5),markerfacecolor='w',markeredgecolor="black", markersize=4),
            plt.Line2D([0], [0], marker='o', color='w', label='p<%s ***'%smartFloat2Str(thresh/50,5),markerfacecolor='w',markeredgecolor="black", markersize=5),
        ]
    else:
        return [
            plt.Line2D([],[],color= COLORS_STRONG[::-1][(i+1)%len(COLORS_STRONG)],
            label = smartIndex([col])[0],)
            for i, col in enumerate(data.columns) if "obj" not in col]+[
            plt.Line2D([],[],color= "blue", label = control_name),
            plt.Line2D([],[],color= "darkblue",linestyle= "--", label = "Effect thresholds - (*,**,***)"),
            plt.Line2D([],[],color= 'darkred', linestyle= "--", label = "Effect thresholds + (*,**,***)"),
            plt.Line2D([],[],color= "gray", linewidth=4, alpha = 0.2, label = "Low effect"),
        ]
            
def autoPlot(lda_dif_score, best_ssmd, positive_control_signature,ref, sinif = None):
    """Plot generated for the automatic analysis

    Args:
        lda_dif_score (float): Mitoscore
        best_ssmd (pandas.DataFrame): columns: SSMD for best 15 descriptors, rows trial condition + all positive controls
        positive_control_signature (pandas.DataFrame): LDAcpScale score for each positiv control
        ref (str): ref name
        sinif (pandas.DataFrame): columns: sinif for best 15 descriptors, rows trial condition + all positive controls

    Returns:
        matplotlib.Figure: the plot figure
    """
    el = lda_dif_score.index[0]
    cps = positive_control_signature.columns
    difval = min(max(lda_dif_score.values[0],0),1)
    psc_rank = positive_control_signature.copy().fillna(0)
    psc_rank = psc_rank.T.sort_values(by=el,ascending=False).iloc[:min(3,len(cps))]
    psc_rank.index = smartIndex(psc_rank.index)


    to_plot = best_ssmd
    to_plot.columns = ["Best descriptors_%s"%col for col in best_ssmd.columns]
    to_plot.index = smartIndex(to_plot.index)
    to_plot = to_plot.T
    if sinif is not None:
        sinif_to_plot = sinif
        sinif_to_plot = sinif_to_plot.T
        sinif_to_plot.columns = to_plot.columns
        sinif_to_plot.index = to_plot.index
        sinif_to_plot.values[...] = thresh2Val(sinif_to_plot.values[...], {1:5,0.05:10,0.01:20,0.001:40})
    fig = plt.figure(figsize=[16,5.52])
    ax1=fig.add_subplot(121, frameon=False)
    colorStamp(ax1,{smartIndex([el])[0]: {"c-":(str(ref[0]),difval),"c+":[(ind,v) for ind,v in zip(psc_rank.index,psc_rank[el])]}})

    ax2=fig.add_subplot(122, projection='polar')
    handles = generateHandle(to_plot,str(ref[0]),shiny=True)
    # first_legend=ax2.legend(
    #     handles=handles,handlelength = 1.5,fontsize = "small",labelspacing = 0.2,shadow=True,
    #     ncol = 3,loc='upper center', bbox_to_anchor=(0.5, -0.1) )
    
    lim = max(min(np.max(abs(to_plot.values)),6),2)
    spiderplot(to_plot.copy(),sinif=sinif_to_plot,ax=ax2,shiny=True, lim=(-lim,lim),sort_val=True)
    first_legend = ax2.legend_
    ax2.legend(
            handles=handles,handlelength = 1,fontsize = "small",labelspacing = 0.1,shadow=True,
            ncol = 3,loc='upper center', bbox_to_anchor=(0.6, -0.05) )
    for label in ax2.get_xticklabels():
        label._y = 0.04
        label.set_fontsize(8)
    plt.tight_layout()
    ax2.add_artist(first_legend)


    return fig

def colorStamp(ax, auto_dict ={"trial1":{"c-":("DMSO",1), "c+":[("CCCP",0),("titi",2), ("toto",4)]},} ):
    """add "thermometer" color scale to the autoPLot

    Args:
        ax (matplotlib.Axes): autoplot ax
        auto_dict (dict, optional): .
    """
    matplotlib
    N=5
    ax.axis("off")
    ax.set_xlim((-0.2,1.2))
    ax.set_ylim((0,1))
    key = list(auto_dict)[0]
    ax.text(0.5,0.9,key.capitalize(),ha="center",va="top", weight="bold",size="large")
    val = auto_dict[key]["c-"][1]/1
    val_id = int(min(np.ceil(6*val),5))
    thermometer(ax,0,0.7,1,0.05,val)
    adj = ["No overall","Weak overall","Limited overall","Measurable overall", "Substantial overall", "Large overall"]
    msg = "%s effect when compared to %s"%(adj[val_id], auto_dict[key]["c-"][0])
    legend = ax.legend([plt.Line2D([],[],linewidth=0)], [msg], 
    loc='center', borderaxespad=0, handlelength = 0, shadow=True, bbox_to_anchor=(0.5, 0.63))
    legend.get_texts()[0].set_color(matplotlib.colormaps.get("turbo")(0.5+val/2))
    ax.add_artist(legend)


    if "c+" in auto_dict[key] and len(auto_dict[key]["c+"])>0:
        adj = ["Strong anti-C","Anti-C","Weak Anti-C","No C","Weak C", "C", "Strong C"]
        contp = auto_dict[key]["c+"]
        y=0.4
        for el, cor in contp:
            thermometer(ax,0.1,y,0.8,0.045,0.1+0.8*(cor+1)/2,cmap="coolwarm",cmap_range=(0,1))
            ax.text(0.5,y+.0225,el,color="w",ha="center",va="center",zorder=2.5)
            y-=0.15

        cp_n = min(3,len(contp))

def thermometer(ax,x,y,w,h, val,border_width = .003, cmap="turbo", cmap_range = (0.5,1)):
    """generate a "thermometer" colorbar with a cursor at a position

    Args:
        ax (matplotlib.Axes): paret ax
        x (float): x of the top left corner of the "thermometer"
        y (float): y of the top left corner of the "thermometer"
        w (float): thrermometer width
        h (float): thrermometer heith
        val (float): cursol position
        border_width (float, optional): border width. Defaults to .003.
        cmap (str, optional): thermometer color map. Defaults to "turbo".
        cmap_range (tuple, optional): thermometer color map range. Defaults to (0.5,1).

    Returns:
        tuple: all the created objects
    """
    bw = border_width
    gradient = np.linspace(*cmap_range, 128)
    gradient = np.vstack((gradient, gradient))
    cx = val
    rect = matplotlib.patches.Rectangle((x-bw, y-bw), w+2*bw, h+2*bw, linewidth=1, facecolor='black',zorder=1 )
    cursor1 = matplotlib.patches.RegularPolygon((cx,y+h), 3, radius=h/3, orientation=np.pi,zorder=3, fc = "black")
    cursor2 = matplotlib.patches.RegularPolygon((cx,y), 3, radius=h/3, orientation=0,zorder=3, fc = "black")
    ax.add_patch(rect)
    ax.add_patch(cursor1)
    ax.add_patch(cursor2)
    im = ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(cmap),vmin=0, vmax=1, extent = (x,x+w,y+h,y),zorder=2)
    return im, rect,cursor1,cursor2

def crossTests(desc_col:pd.Series, type_test = "p_val", hide_low=False,thresh_low=0.05):

    if type_test == "p_val":
        test = lambda x, y:ttest_ind(x,y,equal_var=False,nan_policy="omit")[1]
    elif type_test == "u_val":
        test = lambda x, y:mannwhitneyu(x,y,nan_policy="omit")[1]
    elif type_test == "ssmd":
        test = lambda x, y:(np.nanmean(x) - np.nanmean(y))/(np.nanvar(x)+np.nanvar(y))**0.5
    
    plt.figure("Cross testing %s"%type_test + " on %s"%desc_col.columns[0])

    keys = desc_col.index.unique()
    n=len(keys)
    cross = np.zeros((n,n))
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if j== i:
                cross[j,i] = np.NaN
            else:
                cross[j,i] = test(desc_col.loc[k1].values,desc_col.loc[k2].values)
    
    df_cross = pd.DataFrame(cross,index=keys,columns=keys)
    df_vue = df_cross.copy()
    if hide_low:
        df_vue[df_vue>thresh_low]=np.NaN
    if type_test == "ssmd":
        current_cmap = matplotlib.colormaps["seismic"].copy()
        current_cmap.set_bad(color='lightgray')
        aspect = min(1, 0.4*df_cross.shape[1]/df_cross.shape[0])
        im = plt.imshow(df_vue, cmap=current_cmap, vmin=-7 , vmax=7 , aspect = aspect)
    else:
        current_cmap = matplotlib.colormaps["YlGn"].copy().reversed()
        current_cmap.set_extremes(bad='lightgray', under=current_cmap(0), over=current_cmap(0.9999))
        if df_cross.isna().values.all():
            max_ = 1
            min_ = 10**-4
        else:
            max_ = min(10**(np.ceil(np.log10(np.nanmax(df_vue.values)))),1)
            min_ = np.nanmin(df_vue.values)
            if min_>0: min_ = 10**(np.floor(np.log10(min_)))
            min_ = max_*10**-4 if min_>max_*10**-4 else min_ if min_>max_*10**-10 else max_*10**-10

        aspect = min(1, 0.4*df_cross.shape[1]/df_cross.shape[0])
        im = plt.imshow(df_vue,norm = matplotlib.colors.LogNorm(min_,max_,True),cmap=current_cmap, aspect = aspect)
    plt.colorbar(im, shrink = 0.3)
    plt.yticks(range(len(df_cross.index)),df_cross.index,fontsize=5)
    plt.xticks(range(len(df_cross.index)),[str(i+1) for i in range(len(df_cross.index))],fontsize=5)
    plt.tight_layout()

    return df_cross

