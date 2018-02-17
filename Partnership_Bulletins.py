import pandas as pd
import requests # API request
import pickle
import os
from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat
from keen.client import KeenClient
from datetime import timedelta
from datetime import date
import datetime
import time

# keen API call
os.chdir('/users/jbuckley/Desktop')
with open('Keen_API_credentials.pickle', 'rb') as f:
    Keen_API_credentials = pickle.load(f)

Keen_silo = 'QZ prod'  #Qz app, Qz obsession
projectID = Keen_API_credentials[Keen_silo]['projectID']
readKey = Keen_API_credentials[Keen_silo]['readKey']
keen = KeenClient(project_id=projectID, read_key=readKey)

class bulletin_analysis:
    """
    Class to retrieve Bulletin data for a certain client for a time period and return charts
    
    ex: client = "Accenture"
    ex: bulletin_dict = {'Bulletin_1':['1184288','1181223'], 
                     'Bulletin_2': ['1180851','1184618']}
    """
    def __init__(self, client, bulletin_urls):
        self.client = client
        self.bulletin_dict = bulletin_urls
        pass
    
    def dates(self,start_date,end_date):
        self.start = start_date
        self.end = end_date
        print("Data from "+ start_date + " through "+ end_date)
    
    def SR_creds(self, directory):
        """
        Directory where your SimpleReach API keys are stored
        
        ex: directory = '/Users/jbuckley/Desktop'
        """
        self.SR_dir = directory
        os.chdir(directory)
        SR_creds = pickle.load( open( "SR_credentials.pkl", "rb" ) )
        self.SRTOKEN = SR_creds['SRTOKEN']
        self.SRAPPKEY = SR_creds['SRAPPKEY']
        print("SimpleReach Access granted")
    
    def SR_get_data(self, dashboard='qz',limit=150):
        """
        Retrieves article data from SimpleReach and returns a DataFrame
        
        Kwargs
        dashboard = 'qz','at_work', 'qz_bulletins','at_work_bulletins'
        limit = the max number of articles you want returned
        """
        if dashboard == 'qz':
            board_id = '530e59b3b91c275929001c3b'
        elif dashboard == 'at_work':
            board_id = '59b08189736b79056d00193f'
        elif dashboard == 'qz_bulletins':
            board_id = '5a33ef58736b7983a0000565'
        elif dashboard == 'at_work_bulletins':
            board_id = '5a33f0ff736b79c65e000ba5'
        
        # API call parameters
        parameters = {'board_id': board_id, # MUST INCLUDE
              'day[gte]': self.start, # INCLUDE start date
              'day[lte]': self.end, # INCLUDE end date
              'limit': limit,
              'metric_groups': 'core_data,social_referral_breakouts,social_actions_by_network', 
              #'fields': 'page',
              'group_by': 'content_id', 
              #'tags': tags, 
              'authors': self.client,
              'sort': '-page_views'}
        
        # API keys
        headers = {"SRTOKEN":self.SRTOKEN,
           "SRAPPKEY":self.SRAPPKEY}
        
        # API URL
        URL = 'https://data.simplereach.com/v1/analytics_reports'

        r = requests.get(URL,headers=headers, params=parameters).json()
        x = r['analytics_reports']
        df_SR = pd.DataFrame(x)
        def get_id(url):
            try:
                x = str(url.split("/")[3])
                return(x)
            except:
                pass
        df_SR["article_id"] = df_SR['url'].apply(get_id)
        self.df_SR = df_SR.copy()
        print("DataFrame = self.df_SR")
        def b_cat(url_id):
            try:
                return([k for k, v in self.bulletin_dict.items() if url_id in v][0])
            except:
                pass
        df_bulletin = df_SR.copy()
        df_bulletin['Bulletin'] = df_bulletin['article_id'].apply(b_cat)
        metric_list = ['facebook_actions',
       'facebook_referrals', 'googleplus_actions', 'googleplus_referrals',
       'linkedin_actions', 'linkedin_referrals', 'page_views',
       'pinterest_actions', 'pinterest_referrals', 'reddit_referrals',
       'social_actions', 'stumbleupon_actions', 'stumbleupon_referrals',
       'time_on_site_total', 'total_engaged_time', 'twitter_actions',
       'twitter_followers', 'twitter_referrals']
        df_bulletin['time_on_site_total'] = df_bulletin['time_on_site_total'].astype(float)
        df_bulletin['total_engaged_time'] = df_bulletin['total_engaged_time'].astype(float)
        df_bulletin = df_bulletin.groupby(['Bulletin'],as_index=False)[metric_list].sum()
        df_bulletin['avg_engaged_time'] = df_bulletin['total_engaged_time'] / df_bulletin['page_views']
        self.df_bulletin = df_bulletin.copy()
        print("(Rows, Columns) : " + str(self.df_SR.shape))
        print("Overall Bulletin Data")
        return(self.df_bulletin)
    
    def plotting_headlines(self, metric1='page_views',metric2='avg_engaged_time',metric3='social_actions',metric4='facebook_referrals',
                     m1_benchmark=0, m2_benchmark=0, m3_benchmark=0, m4_benchmark=0, 
                     font='Adele Sans', ylabel='ylabel', xlabel='xlabel', 
                     bar_color='red', bench_color='black'):
        """
        Plots Partnership Data for Bulletins
        """
        # FONTS
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = [font]
        rcParams['font.serif'] = [font]
        title_font = {'size':'24', 'color':'#404347', 'weight':'normal'}
        font_name = {'fontname':font}
        
        # PLOT CREATION
        fig = plt.figure(num=None, figsize=(8, 6), dpi=120, facecolor='#fcfcfc', edgecolor='k') 
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        
        # All X values
        x_values = self.df_SR['title']
        xlabel = "Headlines"
        # All Y values
        y_1 = self.df_SR[metric1]
        y_2 = self.df_SR[metric2]
        y_3 = self.df_SR[metric3]
        y_4 = self.df_SR[metric4]
        
        # PLOT #1
        x_vals = x_values
        x_labels = [y[:15]+("...") for y in x_values]
        y_vals = y_1
        N = len(x_values)

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35       # the width of the bars


        
        rects1 = ax1.bar(ind, y_vals, width, color=bar_color)
        x = np.linspace(-.5, len(x_values), 50)
        if m1_benchmark > 0:
            bench = np.linspace(m1_benchmark, m1_benchmark, 50)
            line1, = ax1.plot(x, bench, '--', linewidth=2, color=bench_color)
        else:
            pass

        # add some text for labels, title and axes ticks
        ax1.set_ylabel(metric1)
        ax1.set_xlabel(xlabel,wrap=True)
        ax1.set_title(metric1, title_font,**font_name)
        ax1.set_xticks(ind + width / 2)
        ax1.set_xticklabels(x_labels, rotation=20,va='top')

        if m1_benchmark > 0:
            ax1.legend((rects1[0],line1), (self.client,'Benchmark'))
        else:
            pass
        
        # PLOT #2
        x_vals = x_values
        x_labels = [y[:15]+("...") for y in x_values]
        y_vals = y_2
        N = len(x_values)

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35       # the width of the bars


        
        rects1 = ax2.bar(ind, y_vals, width, color=bar_color)
        x = np.linspace(-.5, len(x_values), 50)
        if m2_benchmark > 0:
            bench = np.linspace(m2_benchmark, m2_benchmark, 50)
            line1, = ax2.plot(x, bench, '--', linewidth=2, color=bench_color)
        else:
            pass

        # add some text for labels, title and axes ticks
        ax2.set_ylabel(metric2)
        ax2.set_xlabel(xlabel,wrap=True)
        ax2.set_title(metric2, title_font,**font_name)
        ax2.set_xticks(ind + width / 2)
        ax2.set_xticklabels(x_labels, rotation=20,va='top')

        if m2_benchmark > 0:
            ax2.legend((rects1[0],line1), (self.client,'Benchmark'))
        else:
            pass
        
        # PLOT #3
        x_vals = x_values
        x_labels = [y[:15]+("...") for y in x_values]
        y_vals = y_3
        N = len(x_values)

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35       # the width of the bars


        
        rects1 = ax3.bar(ind, y_vals, width, color=bar_color)
        x = np.linspace(-.5, len(x_values), 50)
        if m3_benchmark > 0:
            bench = np.linspace(m3_benchmark, m3_benchmark, 50)
            line1, = ax3.plot(x, bench, '--', linewidth=2, color=bench_color)
        else:
            pass

        # add some text for labels, title and axes ticks
        ax3.set_ylabel(metric3)
        ax3.set_xlabel(xlabel,wrap=True)
        ax3.set_title(metric3, title_font,**font_name)
        ax3.set_xticks(ind + width / 2)
        ax3.set_xticklabels(x_labels, rotation=20,va='top')

        if m3_benchmark > 0:
            ax3.legend((rects1[0],line1), (self.client,'Benchmark'))
        else:
            pass
        
        # PLOT #4
        x_vals = x_values
        x_labels = [y[:15]+("...") for y in x_values]
        y_vals = y_4
        N = len(x_values)

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35       # the width of the bars


        
        rects1 = ax4.bar(ind, y_vals, width, color=bar_color)
        x = np.linspace(-.5, len(x_values), 50)
        if m4_benchmark > 0:
            bench = np.linspace(m4_benchmark, m4_benchmark, 50)
            line1, = ax4.plot(x, bench, '--', linewidth=2, color=bench_color)
        else:
            pass

        # add some text for labels, title and axes ticks
        ax4.set_ylabel(metric4)
        ax4.set_xlabel(xlabel,wrap=True)
        ax4.set_title(metric4, title_font,**font_name)
        ax4.set_xticks(ind + width / 2)
        ax4.set_xticklabels(x_labels, rotation=20,va='top')

        if m4_benchmark > 0:
            ax4.legend((rects1[0],line1), (self.client,'Benchmark'))
        else:
            pass
        
        plt.tight_layout()
        
    def plotting_bulletins(self, metric1='page_views',metric2='avg_engaged_time',metric3='social_actions',metric4='facebook_referrals',
                     m1_benchmark=0, m2_benchmark=0, m3_benchmark=0, m4_benchmark=0, 
                     font='Adele Sans', ylabel='ylabel', xlabel='xlabel', 
                     bar_color='red', bench_color='black'):
        """
        Plots Partnership Data for Bulletins
        """
        # FONTS
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = [font]
        rcParams['font.serif'] = [font]
        title_font = {'size':'24', 'color':'#404347', 'weight':'normal'}
        font_name = {'fontname':font}
        
        # PLOT CREATION
        fig = plt.figure(num=None, figsize=(8, 6), dpi=120, facecolor='#fcfcfc', edgecolor='k') 
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        
        # All X values
        x_values = self.df_bulletin['Bulletin']
        xlabel = "Headlines"
        # All Y values
        y_1 = self.df_bulletin[metric1]
        y_2 = self.df_bulletin[metric2]
        y_3 = self.df_bulletin[metric3]
        y_4 = self.df_bulletin[metric4]
        
        # PLOT #1
        x_vals = x_values
        x_labels = [y[:15]+("...") for y in x_values]
        y_vals = y_1
        N = len(x_values)

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35       # the width of the bars


        
        rects1 = ax1.bar(ind, y_vals, width, color=bar_color)
        x = np.linspace(-.5, len(x_values), 50)
        if m1_benchmark > 0:
            bench = np.linspace(m1_benchmark, m1_benchmark, 50)
            line1, = ax1.plot(x, bench, '--', linewidth=2, color=bench_color)
        else:
            pass

        # add some text for labels, title and axes ticks
        ax1.set_ylabel(metric1)
        ax1.set_xlabel(xlabel,wrap=True)
        ax1.set_title(metric1, title_font,**font_name)
        ax1.set_xticks(ind + width / 2)
        ax1.set_xticklabels(x_labels, rotation=20,va='top')

        if m1_benchmark > 0:
            ax1.legend((rects1[0],line1), (self.client,'Benchmark'))
        else:
            pass
        
        # PLOT #2
        x_vals = x_values
        x_labels = [y[:15]+("...") for y in x_values]
        y_vals = y_2
        N = len(x_values)

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35       # the width of the bars


        
        rects1 = ax2.bar(ind, y_vals, width, color=bar_color)
        x = np.linspace(-.5, len(x_values), 50)
        if m2_benchmark > 0:
            bench = np.linspace(m2_benchmark, m2_benchmark, 50)
            line1, = ax2.plot(x, bench, '--', linewidth=2, color=bench_color)
        else:
            pass

        # add some text for labels, title and axes ticks
        ax2.set_ylabel(metric2)
        ax2.set_xlabel(xlabel,wrap=True)
        ax2.set_title(metric2, title_font,**font_name)
        ax2.set_xticks(ind + width / 2)
        ax2.set_xticklabels(x_labels, rotation=20,va='top')

        if m2_benchmark > 0:
            ax2.legend((rects1[0],line1), (self.client,'Benchmark'))
        else:
            pass
        
        # PLOT #3
        x_vals = x_values
        x_labels = [y[:15]+("...") for y in x_values]
        y_vals = y_3
        N = len(x_values)

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35       # the width of the bars


        
        rects1 = ax3.bar(ind, y_vals, width, color=bar_color)
        x = np.linspace(-.5, len(x_values), 50)
        if m3_benchmark > 0:
            bench = np.linspace(m3_benchmark, m3_benchmark, 50)
            line1, = ax3.plot(x, bench, '--', linewidth=2, color=bench_color)
        else:
            pass

        # add some text for labels, title and axes ticks
        ax3.set_ylabel(metric3)
        ax3.set_xlabel(xlabel,wrap=True)
        ax3.set_title(metric3, title_font,**font_name)
        ax3.set_xticks(ind + width / 2)
        ax3.set_xticklabels(x_labels, rotation=20,va='top',ha='right')

        if m3_benchmark > 0:
            ax3.legend((rects1[0],line1), (self.client,'Benchmark'))
        else:
            pass
        
        # PLOT #4
        x_vals = x_values
        x_labels = [y[:15]+("...") for y in x_values]
        y_vals = y_4
        N = len(x_values)

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35       # the width of the bars


        
        rects1 = ax4.bar(ind, y_vals, width, color=bar_color)
        x = np.linspace(-.5, len(x_values), 50)
        if m4_benchmark > 0:
            bench = np.linspace(m4_benchmark, m4_benchmark, 50)
            line1, = ax4.plot(x, bench, '--', linewidth=2, color=bench_color)
        else:
            pass

        # add some text for labels, title and axes ticks
        ax4.set_ylabel(metric4)
        ax4.set_xlabel(xlabel,wrap=True)
        ax4.set_title(metric4, title_font,**font_name)
        ax4.set_xticks(ind + width / 2)
        ax4.set_xticklabels(x_labels, rotation=20,va='top')

        if m4_benchmark > 0:
            ax4.legend((rects1[0],line1), (self.client,'Benchmark'))
        else:
            pass
        
        plt.tight_layout()

    def interactive_sessions(self,start,end,client):
        """
        Keen API call to collect interactive sessions for bulletin content
        """
        event = 'ad_interaction'

        timeframe = {'start':start, 'end':end}
        interval = None
        timezone = None
        target_property = 'user.cookie.session.id'

        group_by = ('ad_meta.client.name', 'ad_meta.campaign.name','ad_meta.creative.name','raw_url','glass.device') # could potentially use 'article.permalink' instead of id

        property_name1 = 'ad_meta.unit.type'
        operator1 = 'eq'
        property_value1 = 'content'

        property_name2 = 'ad_meta.client.name'
        operator2 = 'contains'
        property_value2 = client


        filters = [{"property_name":property_name1, "operator":operator1, "property_value":property_value1},
                  {"property_name":property_name2, "operator":operator2, "property_value":property_value2}]

        data = keen.count_unique(event, 
                        target_property=target_property,
                        timeframe=timeframe,
                        interval=interval,
                        timezone=timezone,
                        group_by=group_by,
                        filters=filters)

        x = pd.DataFrame.from_dict(data)
        x['start'] = start
        return x

    def timeframe_gen(self, thread_interval=24, tz='US/Eastern'):
        """creates timeframe for use in making Keen API calls
        + args
        start - start date (str - '2017-08-04'); inclusive
        end - end date (str - '2017-12-04'); inclusive; it will always include
            and never exceed this date
        + kwargs:
        hour_interval - interval for breaking up start, end tuple
        tz - timezone, default to US/Eastern

        returns:
            List of tuples; tuple - (start, end)
        """
        freq = str(thread_interval) + 'H'
        end = pd.to_datetime(self.end)
        end = end + datetime.timedelta(1)
        start_dates = pd.date_range(self.start, end, freq=freq, tz=tz)
        start_dates = start_dates.tz_convert('UTC')
        end_dates = start_dates + pd.Timedelta(1, unit='D')

        start_times = [datetime.datetime.strftime(
            i, '%Y-%m-%dT%H:%M:%S.000Z') for i in start_dates]
        end_times = [datetime.datetime.strftime(
            i, '%Y-%m-%dT%H:%M:%S.000Z') for i in end_dates]
        
        self.start_times = start_times[:-1]
        self.end_times = end_times[:-1]
        return(start_times[:-1], end_times[:-1])
    
    def data_calls(self, pools=8):
        """
        Takes 4 arguments: 
        - A list of start times
        - A list of end times 
        - A client/advertiser. 
        - Number of threading pools

        Returns:
        - Time spent data
        - Pageview data
        - Click data
        - Unique page view data
        - Interactive session data

        """
        pool = ThreadPool(pools) 
        start_time = time.time()
  
        iter_pool = [(self.start_times[i], self.end_times[i], self.client.lower()) for i in range(len(self.start_times))]
        int_sessions = pool.starmap(self.interactive_sessions, iter_pool)

        #int_sessions = pool.starmap(self.interactive_sessions, zip(self.start_times, self.end_times, self.client))
        print('Interactive sessions: Done')
        # close the pool and wait for the work to finish 
        pool.close() 
        pool.join() 
        print(str(pools), "Pools took", time.time() - start_time, "to run")
        self.df_int = pd.concat(int_sessions)
        return(self.df_int)
