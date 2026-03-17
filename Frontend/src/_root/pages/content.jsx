// PostsContext.js
import { createContext, useState, useEffect } from "react";

export const PostsContext = createContext();

export const PostsProvider = ({ children }) => {
  // Load initial posts from localStorage
  const [posts, setPosts] = useState(() => {
    const saved = localStorage.getItem("deepfake_posts");
    return saved ? JSON.parse(saved) : [];
  });

  // Save to localStorage whenever posts change
  useEffect(() => {
    try {
      if (posts.length > 0) {
        localStorage.setItem("deepfake_posts", JSON.stringify(posts));
      }
    } catch (error) {
      console.error("Local storage save failed:", error);
      if (error.name === 'QuotaExceededError') {
        // If 10 is still too much (rare), try dropping to 3 as emergency
        const emergencyPrune = posts.slice(-3);
        try {
          localStorage.setItem("deepfake_posts", JSON.stringify(emergencyPrune));
          setPosts(emergencyPrune);
        } catch (e) {
          localStorage.clear();
          setPosts([]);
          alert("Storage limit reached. High-resolution media cleared to restore app.");
        }
      }
    }
  }, [posts]);

  const updateLikes = (postId) => {
    setPosts((prevPosts) =>
      prevPosts.map((post) =>
        post.id === postId ? { ...post, likes: post.likes === 1 ? 0 : 1 } : post
      )
    );
  };

  const addPost = (newPost) => {
    const postWithId = {
      ...newPost,
      id: newPost.id || Date.now(),
      likes: newPost.likes || 0
    };
    
    setPosts((prevPosts) => {
      const nextPosts = [...prevPosts, postWithId];
      // Prune to 10 here to avoid double-render in useEffect
      return nextPosts.length > 10 ? nextPosts.slice(-10) : nextPosts;
    });
  };

  const deletePost = (postId) => {
    setPosts((prevPosts) => prevPosts.filter((post) => post.id !== postId));
  };

  return (
    <PostsContext.Provider value={{ posts, addPost, updateLikes, deletePost }}>
      {children}
    </PostsContext.Provider>
  );
};

