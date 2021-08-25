import React from 'react';
import { Theme, createStyles, makeStyles } from '@material-ui/core/styles';
import ImageList from '@material-ui/core/ImageList';
import ImageListItem from '@material-ui/core/ImageListItem';
import ImageListItemBar from '@material-ui/core/ImageListItemBar';
import IconButton from '@material-ui/core/IconButton';
import StarBorderIcon from '@material-ui/icons/StarBorder';
//import itemData from './ItemData';

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    root: {
      display: 'flex',
      flexWrap: 'wrap',
      justifyContent: 'space-around',
      overflow: 'hidden',
      backgroundColor: theme.palette.background.paper,
    },
    imageList: {
        flexWrap: 'nowrap',
        // Promote the list into his own layer on Chrome. This cost memory but helps keeping high FPS.
        transform: 'translateZ(0)',
    },
    title: {
      color: theme.palette.primary.light,
    },
    titleBar: {
      background:
        'linear-gradient(to top, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.3) 70%, rgba(0,0,0,0) 100%)',
    },
  }),
);

const itemData =[
    {
        img:('https://www.pixelstalk.net/wp-content/uploads/2016/08/Free-Cool-Photography-Backgrounds-camera.jpg'),
    }, 
    {
        img:('https://images2.fanpop.com/images/photos/3300000/Cool-clouds-sunsets-and-sunrises-3334619-2560-1920.jpg'),
    },
    {
        img:('https://img.xcitefun.net/users/2010/08/209105,xcitefun-cool-nature-wallpapers-3.jpg'),
    },
    {
        img:('https://www.vamosrayos.com/b/2020/01/cool-background-hd-images-download-3d-soccer-for-editing-1080p-scaled.jpg'),
    },
];

export default function PhotoSlider() {
  const classes = useStyles();

  return (
    <div className={classes.root}>
      <ImageList className={classes.imageList}>
        {itemData.map((item) => (
          <ImageListItem key={item.img}>
            {console.log(`Testing ${item.img}`)}
            <img src={item.img} alt='data' />
            <ImageListItemBar
              title={'data'}
              classes={{
                root: classes.titleBar,
                title: classes.title,
              }}
              actionIcon={
                <IconButton aria-label={`star ${'data'}`}>
                  <StarBorderIcon className={classes.title} />
                </IconButton>
              }
            />
          </ImageListItem>
        ))}
      </ImageList>
    </div>
  );
}