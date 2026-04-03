import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('/home/claude/charts', exist_ok=True)

df = pd.read_csv('/mnt/user-data/uploads/Hotel_bookings_final.csv')
for col in ['booking_date','check_in_date','check_out_date','travel_date']:
    df[col] = pd.to_datetime(df[col], errors='coerce')
df['is_cancelled']     = (df['booking_status']=='Cancelled').astype(int)
df['is_confirmed']     = (df['booking_status']=='Confirmed').astype(int)
df['length_of_stay']   = (df['check_out_date']-df['check_in_date']).dt.days
df['profit']           = df['selling_price'] - df['costprice']
df['booking_month']    = df['booking_date'].dt.to_period('M').astype(str)
df['booking_monthnum'] = df['booking_date'].dt.month
df['booking_quarter']  = df['booking_date'].dt.quarter
df['season']           = df['booking_monthnum'].map(lambda m:
    'Summer' if m in [6,7,8] else 'Winter' if m in [12,1,2] else
    'Spring' if m in [3,4,5] else 'Autumn')
confirmed = df[df['booking_status']=='Confirmed'].copy()
cancelled = df[df['booking_status']=='Cancelled'].copy()

BLUE='#1A5F9E'; TEAL='#0D9E75'; ORANGE='#E8592A'; AMBER='#E8A020'
GRAY='#7A7A7A'; LGRAY='#F4F4F4'; DBLUE='#0C3F6B'

def style_ax(ax, title='', ylabel='', xlabel=''):
    ax.set_facecolor(LGRAY)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color='#1a1a1a')
    ax.set_xlabel(xlabel, fontsize=9, color=GRAY)
    ax.set_ylabel(ylabel, fontsize=9, color=GRAY)
    ax.tick_params(colors=GRAY, labelsize=8.5)
    ax.spines[['top','right','left','bottom']].set_visible(False)
    ax.grid(axis='y', color='white', linewidth=1.2, zorder=0)

# CHART 1: Monthly Revenue + Cancel Rate
monthly = df.groupby('booking_month').agg(
    revenue=('selling_price', lambda x: x[df.loc[x.index,'booking_status']=='Confirmed'].sum()),
    cancel_pct=('is_cancelled','mean')
).reset_index()
monthly['cancel_pct'] *= 100
months_short = [m[-5:].replace('-','/') for m in monthly['booking_month']]

fig, ax1 = plt.subplots(figsize=(13,5)); fig.patch.set_facecolor('white')
ax1.set_facecolor(LGRAY)
ax1.bar(months_short, monthly['revenue']/1e6, color=BLUE, alpha=0.85, zorder=3, width=0.6)
ax1.set_ylabel('Revenue (₹ Millions)', fontsize=9, color=GRAY)
ax1.tick_params(colors=GRAY, labelsize=8)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('₹%.0fM'))
ax1.spines[['top','right','left','bottom']].set_visible(False)
ax1.grid(axis='y', color='white', linewidth=1.2, zorder=0)
ax2 = ax1.twinx()
ax2.plot(months_short, monthly['cancel_pct'], color=ORANGE, linewidth=2.5, marker='o', markersize=6, zorder=4, label='Cancel Rate %')
ax2.set_ylabel('Cancellation Rate (%)', fontsize=9, color=ORANGE)
ax2.tick_params(colors=ORANGE, labelsize=8.5)
ax2.spines[['top','right','left','bottom']].set_visible(False)
ax2.set_ylim(0,40)
ax1.set_title('Monthly Revenue vs Cancellation Rate  |  Apr 2024 – Apr 2025', fontsize=13, fontweight='bold', pad=10)
fig.legend(loc='upper right', bbox_to_anchor=(0.97,0.94), fontsize=9)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('/home/claude/charts/chart1_monthly.png', dpi=150, bbox_inches='tight'); plt.close()
print('Chart 1 saved')

# CHART 2: Channel Performance
ch = df.groupby('booking_channel').agg(
    revenue=('selling_price', lambda x: x[df.loc[x.index,'booking_status']=='Confirmed'].sum()),
    avg_profit=('profit','mean'),
    cancel_pct=('is_cancelled','mean')
).reset_index()
ch['cancel_pct'] *= 100
ch['revenue_m'] = ch['revenue']/1e6
colors3 = [BLUE, TEAL, ORANGE]

fig, axes = plt.subplots(1,3, figsize=(13,5)); fig.patch.set_facecolor('white')
fig.suptitle('Booking Channel Performance', fontsize=13, fontweight='bold', color='#1a1a1a')
for ax, col, ylabel, fmt, pal in zip(
    axes,
    ['revenue_m','cancel_pct','avg_profit'],
    ['₹ Millions','Cancellation %','₹ Profit'],
    ['₹{:.0f}M','{:.1f}%','₹{:,.0f}'],
    [colors3, colors3, colors3]
):
    bars = ax.bar(ch['booking_channel'], ch[col], color=pal, zorder=3, width=0.5)
    style_ax(ax, ylabel, ylabel)
    for bar, val in zip(bars, ch[col]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                fmt.format(val), ha='center', fontsize=9, fontweight='bold', color='#1a1a1a')
plt.tight_layout(); plt.savefig('/home/claude/charts/chart2_channels.png', dpi=150, bbox_inches='tight'); plt.close()
print('Chart 2 saved')

# CHART 3: Season Heatmap
season_order = ['Spring','Summer','Autumn','Winter']
sc = df.groupby(['season','booking_channel'])['is_cancelled'].mean().mul(100).unstack().reindex(season_order)
fig, ax = plt.subplots(figsize=(8,4)); fig.patch.set_facecolor('white')
im = ax.imshow(sc.values, cmap='RdYlGn_r', aspect='auto', vmin=10, vmax=35)
ax.set_xticks(range(len(sc.columns))); ax.set_xticklabels(sc.columns, fontsize=11)
ax.set_yticks(range(len(sc.index))); ax.set_yticklabels(sc.index, fontsize=11)
ax.set_title('Cancellation Rate: Season × Channel (%)', fontsize=12, fontweight='bold', pad=10)
for i in range(len(sc.index)):
    for j in range(len(sc.columns)):
        val = sc.values[i,j]
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=13,
                fontweight='bold', color='white' if val>27 else '#1a1a1a')
plt.colorbar(im, ax=ax, label='Cancel %', shrink=0.85)
ax.spines[['top','right','left','bottom']].set_visible(False)
plt.tight_layout(); plt.savefig('/home/claude/charts/chart3_heatmap.png', dpi=150, bbox_inches='tight'); plt.close()
print('Chart 3 saved')

# CHART 4: Room Type
rt = confirmed.groupby('room_type').agg(
    bookings=('booking_status','count'),
    avg_selling=('selling_price','mean'),
    avg_profit=('profit','mean')
).reset_index()
fig, axes = plt.subplots(1,2, figsize=(12,5)); fig.patch.set_facecolor('white')
fig.suptitle('Room Type Performance', fontsize=13, fontweight='bold')
x=np.arange(len(rt)); w=0.38
ax=axes[0]
b1=ax.bar(x-w/2, rt['avg_selling'], w, label='Avg Price', color=BLUE, zorder=3)
b2=ax.bar(x+w/2, rt['avg_profit'], w, label='Avg Profit', color=TEAL, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(rt['room_type'])
style_ax(ax,'Avg Price vs Profit by Room Type','₹')
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x:,.0f}'))
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100, f'₹{bar.get_height():,.0f}', ha='center', fontsize=7.5, color='#1a1a1a', fontweight='bold')
ax=axes[1]
wedges,texts,autos=ax.pie(rt['bookings'], labels=rt['room_type'], autopct='%1.1f%%',
    colors=[BLUE,TEAL,ORANGE], explode=[0,0,0.05], startangle=140, textprops={'fontsize':11})
for at in autos: at.set_fontsize(10); at.set_color('white'); at.set_fontweight('bold')
ax.set_title('Booking Share by Room Type', fontsize=11, fontweight='bold', pad=10)
plt.tight_layout(); plt.savefig('/home/claude/charts/chart4_roomtype.png', dpi=150, bbox_inches='tight'); plt.close()
print('Chart 4 saved')

# CHART 5: City Revenue
city = confirmed.groupby('city').agg(revenue=('selling_price','sum'), avg_profit=('profit','mean')).reset_index().sort_values('revenue')
fig, ax = plt.subplots(figsize=(10,6)); fig.patch.set_facecolor('white')
colors_city = [BLUE if r==city['revenue'].max() else TEAL if r>=city['revenue'].quantile(0.5) else GRAY for r in city['revenue']]
bars = ax.barh(city['city'], city['revenue']/1e6, color=colors_city, zorder=3, height=0.6)
ax.set_facecolor(LGRAY); ax.spines[['top','right','left','bottom']].set_visible(False)
ax.grid(axis='x', color='white', linewidth=1.2, zorder=0)
ax.set_title('Total Revenue by City', fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('₹ Millions', fontsize=9, color=GRAY)
ax.tick_params(colors=GRAY, labelsize=9)
for bar, val in zip(bars, city['revenue']/1e6):
    ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2, f'₹{val:.1f}M', va='center', fontsize=9, fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('₹%.0fM'))
plt.tight_layout(); plt.savefig('/home/claude/charts/chart5_city.png', dpi=150, bbox_inches='tight'); plt.close()
print('Chart 5 saved')

# CHART 6: Quarterly
qdata = df.groupby('booking_quarter').agg(
    revenue=('selling_price', lambda x: x[df.loc[x.index,'booking_status']=='Confirmed'].sum()),
    cancel_pct=('is_cancelled','mean')
).reset_index()
qdata['cancel_pct']*=100; qdata['rev_m']=qdata['revenue']/1e6
qlabels=['Q1\n(Jan-Mar)','Q2\n(Apr-Jun)','Q3\n(Jul-Sep)','Q4\n(Oct-Dec)']
fig, axes=plt.subplots(1,2, figsize=(11,5)); fig.patch.set_facecolor('white')
fig.suptitle('Quarterly Analysis', fontsize=13, fontweight='bold')
colors_q=[BLUE,TEAL,ORANGE,AMBER]
ax=axes[0]
bars=ax.bar(qlabels, qdata['rev_m'], color=colors_q, zorder=3, width=0.5)
style_ax(ax,'Revenue by Quarter','₹ Millions')
for bar,val in zip(bars,qdata['rev_m']): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'₹{val:.0f}M', ha='center', fontsize=9.5, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('₹%.0fM'))
ax=axes[1]
bars=ax.bar(qlabels, qdata['cancel_pct'], color=colors_q, zorder=3, width=0.5)
style_ax(ax,'Cancellation Rate by Quarter','%')
for bar,val in zip(bars,qdata['cancel_pct']): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold', color=ORANGE)
ax.axhline(y=20.2, color=GRAY, linestyle='--', linewidth=1.5, label='Avg'); ax.legend(fontsize=9); ax.set_ylim(0,35)
plt.tight_layout(); plt.savefig('/home/claude/charts/chart6_quarterly.png', dpi=150, bbox_inches='tight'); plt.close()
print('Chart 6 saved')

# CHART 7: Projection
monthly_vals = monthly['revenue'].values/1e6
x_hist=np.arange(len(monthly_vals))
z=np.polyfit(x_hist, monthly_vals, 1); p_fn=np.poly1d(z)
x_proj=np.arange(len(monthly_vals), len(monthly_vals)+3)
proj_vals=p_fn(x_proj)
proj_months=['05/25','06/25','07/25']
fig, ax=plt.subplots(figsize=(13,5)); fig.patch.set_facecolor('white')
ax.set_facecolor(LGRAY)
ax.fill_between(range(len(monthly_vals)), monthly_vals, alpha=0.12, color=BLUE)
ax.plot(range(len(monthly_vals)), monthly_vals, color=BLUE, linewidth=2.5, marker='o', markersize=7, label='Actual', zorder=4)
ax.plot(list(range(len(monthly_vals)-1, len(monthly_vals)+3)), [monthly_vals[-1]]+list(proj_vals),
    color=ORANGE, linewidth=2.5, marker='s', markersize=7, linestyle='--', label='Projected', zorder=4)
ax.axvspan(len(monthly_vals)-0.5, len(monthly_vals)+2.5, alpha=0.07, color=ORANGE)
all_m=months_short+proj_months
ax.set_xticks(range(len(all_m))); ax.set_xticklabels(all_m, rotation=30, ha='right', fontsize=8.5)
ax.set_title('Revenue Trend & 3-Month Projection (Linear Trend)', fontsize=13, fontweight='bold', pad=10)
ax.set_ylabel('₹ Millions', fontsize=9, color=GRAY)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('₹%.0fM'))
ax.tick_params(colors=GRAY); ax.spines[['top','right','left','bottom']].set_visible(False)
ax.grid(axis='y', color='white', linewidth=1.2, zorder=0)
for i,val in enumerate(proj_vals):
    ax.annotate(f'₹{val:.1f}M', (len(monthly_vals)+i, val), xytext=(0,12), textcoords='offset points',
                ha='center', fontsize=9, color=ORANGE, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout(); plt.savefig('/home/claude/charts/chart7_projection.png', dpi=150, bbox_inches='tight'); plt.close()
print('Chart 7 saved')

# CHART 8: Star rating
star=df.groupby('star_rating').agg(avg_price=('selling_price','mean'), avg_profit=('profit','mean'), cancel_pct=('is_cancelled','mean')).reset_index()
star['cancel_pct']*=100
fig, axes=plt.subplots(1,2, figsize=(11,5)); fig.patch.set_facecolor('white')
fig.suptitle('Star Rating Analysis', fontsize=13, fontweight='bold')
slabels=[f'{s}★' for s in star['star_rating']]
colors_s=[LGRAY,TEAL,BLUE,DBLUE]
ax=axes[0]
bars=ax.bar(slabels, star['avg_price'], color=colors_s, zorder=3, width=0.5)
style_ax(ax,'Avg Selling Price by Star Rating','₹')
for bar,val in zip(bars,star['avg_price']): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50, f'₹{val:,.0f}', ha='center', fontsize=9, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x:,.0f}'))
ax=axes[1]
colors_c=[TEAL if v<21 else ORANGE for v in star['cancel_pct']]
bars=ax.bar(slabels, star['cancel_pct'], color=colors_c, zorder=3, width=0.5)
style_ax(ax,'Cancellation Rate by Star Rating','%')
for bar,val in zip(bars,star['cancel_pct']): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold', color=ORANGE)
ax.axhline(y=20.2, color=GRAY, linestyle='--', linewidth=1.5, label='Avg 20.2%'); ax.legend(fontsize=9); ax.set_ylim(0,30)
plt.tight_layout(); plt.savefig('/home/claude/charts/chart8_starrating.png', dpi=150, bbox_inches='tight'); plt.close()
print('Chart 8 saved')

print('\n=== KEY NUMBERS ===')
print(f'Total bookings: {len(df):,}')
print(f'Confirmed: {df.is_confirmed.sum():,} | Cancelled: {df.is_cancelled.sum():,} | Failed: {(df.booking_status=="Failed").sum():,}')
print(f'Total Revenue (confirmed): Rs {confirmed.selling_price.sum()/1e6:.1f}M')
print(f'Total Profit: Rs {confirmed.profit.sum()/1e6:.1f}M  |  Avg/booking: Rs {confirmed.profit.mean():,.0f}')
print(f'Revenue at risk (cancellations): Rs {cancelled.booking_value.sum()/1e6:.1f}M')
print(f'Travel Agent cancel: {df[df.booking_channel=="Travel Agent"].is_cancelled.mean()*100:.1f}%  |  Web: {df[df.booking_channel=="Web"].is_cancelled.mean()*100:.1f}%')
print(f'Summer cancel: {df[df.season=="Summer"].is_cancelled.mean()*100:.1f}%  |  Spring: {df[df.season=="Spring"].is_cancelled.mean()*100:.1f}%')
print(f'Q3 cancel: {df[df.booking_quarter==3].is_cancelled.mean()*100:.1f}%  |  Q2: {df[df.booking_quarter==2].is_cancelled.mean()*100:.1f}%')
print(f'Top city: {confirmed.groupby("city").selling_price.sum().idxmax()}')
print(f'Suite avg profit: Rs {confirmed[confirmed.room_type=="Suite"].profit.mean():,.0f}')
print(f'Projected May-Jul 2025: Rs {proj_vals.sum():.1f}M')
print('\nAll charts saved.')
