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
      // KEEPING IT CLEAN: Automatically prune to top 10 most recent posts
      // to avoid QuotaExceededError while still showing recent results.
      let postsToSave = [...posts];
      
      if (postsToSave.length > 10) {
        postsToSave = postsToSave.slice(postsToSave.length - 10);
        // We update the state too so the UI stays in sync with what's actually saved
        setPosts(postsToSave);
      }

      localStorage.setItem("deepfake_posts", JSON.stringify(postsToSave));
    } catch (error) {
      console.error("Local storage save failed:", error);
      // If still failing after pruning, clear all as emergency measure
      if (error.name === 'QuotaExceededError') {
        localStorage.clear();
        alert("Local storage is full. Cleared old history to maintain stability.");
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
    // Ensure post has a unique ID and initial likes if not present
    const postWithId = {
      ...newPost,
      id: newPost.id || Date.now(),
      likes: newPost.likes || 0
    };
    setPosts((prevPosts) => [...prevPosts, postWithId]);
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

