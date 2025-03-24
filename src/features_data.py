import  src.features_global as glob
from src.external import *
from src.features_stat import Descriptor, DESCRIPTOR_TYPE

class DataManager():
    data: pd.DataFrame
    descriptors: dict
    index: set
    features: list
    controls:set
    droped_cache:list
    def __init__(self) -> None:
        self.data: pd.DataFrame = pd.DataFrame()
        self.descriptors: dict = {}
        self.index: set = set()
        self.features = []
        self.controls:set = set()
        self.droped_cache:list  = []
        self.aggregators:list = []
    
    def set_data(self,data:pd.DataFrame):
        self.data = data
        self.features = data.columns
        self.descriptors = {}
        self.index = {}
        self.controls = set()
        self.solve_descriptors_types()
        self.droped_cache.clear()

    def as_copy(self,data_manager):
        assert type(data_manager) is DataManager         
        self.droped_cache.clear()
        self.descriptors= {}
        self.features = data_manager.features.copy()
        self.data: pd.DataFrame = data_manager.data.reset_index()[self.features]
        self.solve_descriptors_types()
        self.set_index(data_manager.index.copy())
        self.controls:set = data_manager.controls.copy()

    def add_column(self,col:pd.Series):
        assert len(col) == len(self.data), "column and df should be the same size"
        self.data[col.name] = np.nan
        self.data[col.name] = col.values
        if col.name not in self.features:
            self.features = self.features.append(pd.Index([col.name]))
            is_numeric = col.dtype.kind in 'biufc' 
            self.descriptors[col.name] = Descriptor(col.name,DESCRIPTOR_TYPE.NUMERIC if is_numeric else DESCRIPTOR_TYPE.CATEGORY)



    def dump_view(self):
        to_dump = DataManager()
        to_dump.data = self.data
        to_dump.features = self.features
        to_dump.index = self.index
        to_dump.controls = self.controls
        return to_dump

    def drop_data_with_index(self, index_to_drop):
        droped = self.data.loc[index_to_drop]
        self.data.drop(index = index_to_drop, inplace=True)
        droped.reset_index(inplace=True)
        self.droped_cache.append(droped)
        self.controls=set()
    def drop_data_with_querry(self, querry):
        df = self.data.reset_index()
        kept = df.query(querry)
        dropped = df.loc[df.index.difference(kept.index)]
        self.droped_cache.append(dropped)
        self.controls=set()
        self.data = kept.set_index(self.data.index.names)

    def undrop_data(self, n=1):
        if len(self.droped_cache)<1 or n==0:
            return        
        n=len(self.droped_cache) if n<0 else min(n,len(self.droped_cache))
        dropped_list=[]
        for i in range(n):
            dropped = self.droped_cache.pop()
            dropped.set_index(self.data.index.names,inplace=True)
            dropped_list.append(dropped)
        self.data = pd.concat([self.data]+ dropped_list ,axis='index',join='outer').sort_index()
        self.controls=set()

    def get_index_counts(self):
        return self.get_safe_data().index.value_counts(sort=False)
    def get_descriptor(self, key):
        if type(key) is str:
            return self.descriptors[key]    
        elif type(key) is int:
            return self.get_descriptor(self.features[key])    
    def get_clust_ordered_descriptors_names(self, descriptor_names=None):
        """reorder the descriptors according to clusters
        """
        
        order = np.argsort([desc.clustId("zz") for desc in self.descriptors.values()] )
        unordered_names = [desc_name for desc_name in self.descriptors.keys()]
        if descriptor_names is None:
            return [unordered_names[i] for i in order]
        return [unordered_names[i] for i in order if unordered_names[i] in descriptor_names]

    def bool_feature_to_string_categorycal_feature(self,feature):
        col =  self.data[feature]
        if not col.dtype.kind in 'biufc':
            return
        unique = set(col.unique())
        if not (unique == {0,1}): 
                return
        self.data.loc[:,feature]= col.map({1: feature, 0: '-'}).astype("category")
        self.descriptors[feature].type = DESCRIPTOR_TYPE.CATEGORY
    def solve_descriptors_types(self,forced_category=[]):
        columns = self.data.columns
        for col in columns:
            if col in forced_category or not self.feature_to_numeric(col):
                self.feature_to_category(col)
            self.bool_feature_to_string_categorycal_feature(col)

    def feature_to_numeric(self,feature):
        data_col = self.data[feature]
        data_col.replace('',np.NaN)
        is_numeric = data_col.dtype.kind in 'biufc' 
        if not is_numeric:
            data_col = pd.to_numeric(data_col, errors="ignore")
        is_numeric = data_col.dtype.kind in 'biufc'
        if not is_numeric:
            return False
        desc_type = DESCRIPTOR_TYPE.NUMERIC
        self.data[feature] = data_col
        if feature in self.descriptors:
            self.descriptors[feature].type = desc_type
        else:
            self.descriptors[feature] = Descriptor(feature,desc_type)
        self.bool_feature_to_string_categorycal_feature(feature)
        return True
    def feature_to_category(self,feature):
        data_col = self.data[feature]
        data_col = data_col.astype("string")
        data_col.fillna("", inplace=True )
        data_col = data_col.astype("category")
        desc_type = DESCRIPTOR_TYPE.CATEGORY
        self.data[feature] = data_col
        if feature in self.descriptors:
            self.descriptors[feature].type = desc_type
        else:
            self.descriptors[feature] = Descriptor(feature,desc_type)
    def get_index_cols(self):
        return [f for f in self.features if f in self.index]

    def get_safe_data(self,allow_category = False):
        data = (self.data if len(self.aggregators)<1 else self.get_data_aggregate() )[self.get_valid_descriptor_names(allow_category)]
        return data.sort_index()

    def get_data_aggregate(self):
        data = self.data.copy()
        index_aggreg = [el[0] for el in self.aggregators ]
        for el in index_aggreg: 
            if el not in self.index:
                data.set_index(el, inplace=True, append=True)
        data = data.groupby(by=data.index.names).aggregate("mean").dropna(how='all')
        return data.reset_index().set_index(self.get_index_cols())
    
    def get_unique_index(self):
        return self.data.index.unique()

    def get_control_index(self):
        unique = self.get_unique_index()
        return [unique[el] for el in  self.controls]
            
    def set_index(self, index_cols=None):
        if index_cols is None:
            index_cols = self.get_index_cols()
        self.index = {col for col in index_cols}
        self.data= self.data.reset_index()[self.features].copy()
        self.data.set_index(self.get_index_cols(),inplace=True)
        self.data.sort_index(inplace=True)

    def get_valid_descriptor_names(self,allow_category = False,clust_orderd=False):
        valid = [f for f in self.features if self.descriptors[f].keep(allow_category) and not(f in self.index)]
        if not clust_orderd:
            return valid
        else: 
            return self.get_clust_ordered_descriptors_names(valid)
    
    def get_valid_descriptor(self,allow_category = False):
        return {k:d for k, d in self.descriptors.items() if d.keep(allow_category) and not(k in self.index)}

    def find_descriptor_corrleations(self, thresh, cor_dist = "square", no_plot=True):
        """Group descriptor in clusters via hierarchical clustering

        Args:
            thresh (float): All element of a same cluster group have distance (correlation squared) > thresh
            cor_dist (str, optional): distance methode between 2 descriptors:
                -square (correlation squared)
                - abs (absolute of correlation) 
                - default (correlation). Defaults to "square".
            no_plot (bool, optional): if true, no plot is shown to user, else, both correlation dendograme and correlation map are showne. Defaults to True.
        """
        data=self.get_safe_data()
        corr_map = data.corr().fillna(0)
        for i in range(len(corr_map)):
            corr_map.iloc[i,i]=1 
        if cor_dist =="square":
            cor_d = corr_map**2
        elif cor_dist =="abs":
            cor_d = abs(corr_map)
        elif cor_dist == "default":
            cor_d = corr_map
        else:
            cor_d = corr_map**2
        pdist = squareform(1-cor_d)

        linkage = spc.linkage(pdist, method='complete')
        linkage = spc.optimal_leaf_ordering(linkage,pdist)
        if not no_plot:
            plt.figure("Correlation Dendogram")
        dend = spc.dendrogram(linkage, labels=corr_map.columns,orientation="right" ,color_threshold=thresh,no_plot=no_plot)
        
        new_order = dend["leaves"]
        f_clusters = spc.fcluster(linkage,thresh,criterion = "distance")[new_order] #dend["leaves_color_list"]
        val,count = np.unique(f_clusters,return_counts = True)
        count_map = {v:c for v,c in zip(val,count)}
        c_names = {};c_it = 1
        for el in f_clusters:
            if count_map[el]>1 and el not in c_names:
                c_names[el]="C%s (%d)"%(str(c_it).zfill(2),count_map[el])
                c_it+=1
        f_clusters = [c_names[el] if el in c_names else "C0" for el in f_clusters]
        f_clusters_dict = {}
        for i, el in enumerate([[desc for desc in self.descriptors.values() if desc.keep()][o] for o in new_order]):
            clust = f_clusters[i]
            if clust !="C0":
                if clust not in f_clusters_dict:
                    f_clusters_dict[clust] = {clust:[el]}                    
                else:
                    f_clusters_dict[clust][clust].append(el)
                el.setCluster(f_clusters_dict[clust])
            else:
                el.setCluster({"C0":[el]})
        for el in [desc for desc in self.descriptors.values()  if not desc.keep()]:
            el.setCluster({"C0":[el]})
        cor_d = cor_d[cor_d.columns[new_order]].T[cor_d.columns[new_order]]
        return cor_d

    def descriptor_corrleations_filtering(self, inplace=True):
        """Only keep one descriptor in each cluster (filtering the others) using anova on wholle dataset to discriminate between descriptors.
        descriptors with the lowest anova in each cluster are kept

        Args:
            inplace (bool, optional): if true filter descriptors and return the list of kept descriptors, else only return the list of kept descriptors.
                Defaults to True.

        Returns:
            list(ana_mod.Descriptor): list of kept descriptors
        """
        data = self.get_safe_data().dropna()
        if not data.empty:
            anova_pval = pd.DataFrame([f_oneway(*[data.loc[ind] for ind in data.index.unique()])[1]],columns = data.columns)
        else: 
            anova_pval = pd.DataFrame([[0 for _ in data.columns]],columns = data.columns)
        filtered = []
        for desc in self.get_valid_descriptor().values() :
            if desc.type!=DESCRIPTOR_TYPE.NUMERIC:
                continue
            if desc.keep() and len(desc.cluster[desc.clustId("C0")])>1:
                c_id = desc.clustId("C0")
                descs = desc.cluster[c_id]
                if sum([d.keep() for d in descs])<2:
                    continue
                keys = [d.getKey() for d in descs if d.keep()]
                anova_pval_local = anova_pval[keys]
                min_ = np.min(anova_pval_local.values)
                for d in descs: 
                    if d.keep() and anova_pval_local[d.getKey()].loc[0]>min_:
                        d.setIgnore(True)
                        filtered.append(d)
                    elif d.keep() and anova_pval_local[d.getKey()].loc[0] == min_: #force choice in case of 2 el cluster
                        min_-=1
        best_desc = {k:d for k,d in self.get_valid_descriptor().items()}
        if not inplace:
            for d in filtered:
                d.setIgnore(False)
        return best_desc
        
    def reset_descriptor_corrleations(self):
        [desc.setCluster({"C0":[desc]}) for desc in self.descriptors.values()]
